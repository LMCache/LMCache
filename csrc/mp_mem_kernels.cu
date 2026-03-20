// SPDX-License-Identifier: Apache-2.0

#include "mp_mem_kernels.cuh"

#include <cstring>

namespace {

/**
 * Key logic in the kernel implementation:
 * 1 Each thread block is for (BS, NH, HS) part (i.e., a single block in the
 * paged buffer) 2 Within a thread block, each warp is for a single head. Number
 * of warps in a thread block is equal to the number of heads (NH). 3 Within a
 * thread block, we do the loop over the BS (i.e., number of tokens in the
 * block) dimension. 4 The grid will take over (2, NB, NL) dimensions. No matter
 * what the actual layout in memory is, we will calculate the global offset for
 * the start of the block 5 For LMCache, we assume it is always using 2LTD
 * layout, e.g., [2, L, 256, NH * HS], where 256 means that 256 tokens
 */

template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_engine_global_offset(
    const int k_or_v, const int engine_block_idx, const int layer_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_block = shape_desc.scalars_per_block<ScalarType>();
  if constexpr (format == GPUKVFormat::NB_NL_TWO_BS_NH_HS) {
    // Cross-layer: single tensor [NB, NL, 2, BS, NH, HS]
    return k_or_v * scalars_per_block +
           layer_idx * shape_desc.kv_size * scalars_per_block +
           engine_block_idx * shape_desc.kv_size * scalars_per_block *
               shape_desc.nl;
  } else if constexpr (format == GPUKVFormat::NL_X_TWO_NB_BS_NH_HS) {
    // Normal: L tensors [2, NB, BS, NH, HS]
    return engine_block_idx * scalars_per_block +
           k_or_v * shape_desc.nb * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    // Flash Infer: L tensors [NB, 2, BS, NH, HS]
    return engine_block_idx * shape_desc.kv_size * scalars_per_block +
           k_or_v * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::NL_X_NB_BS_HS) {
    // MLA: L tensors [NB, BS, HS]
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS) {
    // SGLang MHA: 2L tensors [NBBS, NH, HS] — K/V via separate tensor ptrs
    return engine_block_idx * scalars_per_block;
  } else if constexpr (format == GPUKVFormat::NL_X_NBBS_ONE_HS) {
    // SGLang MLA: L tensors [NBBS, 1, HS]
    return engine_block_idx * scalars_per_block;
  }
}

template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_engine_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  // for now, everything is BS, NH, HS, so we only have a single case
  return head_idx * scalars_per_head + token_offset * scalars_per_token;
}

template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_lmcache_global_offset(
    const int k_or_v,
    const int
        token_offset_in_lmcache_block,  // 0~255 if LMCache block size is 256
    const int layer_idx,
    const int lmcache_chunk_size,  // e.g., 256
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  // LMCache is using 2LTD all the times
  return token_offset_in_lmcache_block * scalars_per_token +
         layer_idx * lmcache_chunk_size * scalars_per_token +
         k_or_v * shape_desc.nl * lmcache_chunk_size * scalars_per_token;
}

template <typename ScalarType, GPUKVFormat format>
__device__ inline size_t calculate_lmcache_local_offset(
    const int token_offset, const int head_idx,
    const PageBufferShapeDesc shape_desc) {
  size_t scalars_per_head = shape_desc.scalars_per_head<ScalarType>();
  size_t scalars_per_token = shape_desc.scalars_per_token<ScalarType>();
  return head_idx * scalars_per_head + token_offset * scalars_per_token;
}

__device__ inline uint4 ld_cs(const uint4* addr) {
  uint4 val;
  asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
  return val;
}

__device__ inline void st_cs(uint4* addr, uint4 val) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

template <typename ScalarType>
__device__ inline void warp_copy(ScalarType* __restrict__ dst,
                                 const ScalarType* __restrict__ src,
                                 size_t num_elements) {
  int idx = threadIdx.x;
  int stride = blockDim.x;
  if constexpr (std::is_same_v<ScalarType, uint4>) {
    for (size_t i = idx; i < num_elements; i += stride) {
      st_cs(dst + i, ld_cs(src + i));
    }
  } else {
    for (size_t i = idx; i < num_elements; i += stride) {
      dst[i] = src[i];
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine, GPUKVFormat format>
__device__ void multi_layer_block_transfer_single_block(
    ScalarType* __restrict__ lmcache_object,
    ScalarType** __restrict__ paged_buffer_ptrs, const int engine_block_idx,
    const int offset_in_lmcache_block, const PageBufferShapeDesc shape_desc,
    const int lmcache_chunk_size  // e.g., 256, used to calculate global offset
                                  // in LMCache object
) {
  const int head_idx = threadIdx.y;
  const int k_or_v = blockIdx.x;
  const int layer_idx = blockIdx.z;

  const size_t engine_global_offset =
      calculate_engine_global_offset<ScalarType, format>(
          k_or_v, engine_block_idx, layer_idx, shape_desc);
  const size_t lmcache_global_offset =
      calculate_lmcache_global_offset<ScalarType, format>(
          k_or_v, offset_in_lmcache_block, layer_idx, lmcache_chunk_size,
          shape_desc);
  ScalarType* paged_buffer_layer_ptr;
  if constexpr (format == GPUKVFormat::NB_NL_TWO_BS_NH_HS) {
    paged_buffer_layer_ptr = (ScalarType*)paged_buffer_ptrs[0];
  } else if constexpr (format == GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS) {
    // SGLang MHA: ptrs[0..NL-1] = K per layer, ptrs[NL..2NL-1] = V per layer
    paged_buffer_layer_ptr =
        (ScalarType*)paged_buffer_ptrs[k_or_v * shape_desc.nl + layer_idx];
  } else {
    paged_buffer_layer_ptr = (ScalarType*)paged_buffer_ptrs[layer_idx];
  }

  for (int token_offset = 0; token_offset < shape_desc.bs; ++token_offset) {
    const size_t engine_local_offset =
        calculate_engine_local_offset<ScalarType, format>(token_offset,
                                                          head_idx, shape_desc);
    const size_t lmcache_local_offset =
        calculate_lmcache_local_offset<ScalarType, format>(
            token_offset, head_idx, shape_desc);
    ScalarType* engine_ptr =
        paged_buffer_layer_ptr + engine_global_offset + engine_local_offset;
    ScalarType* lmcache_ptr =
        lmcache_object + lmcache_global_offset + lmcache_local_offset;
    if constexpr (lmcache_to_engine) {
      warp_copy<ScalarType>(engine_ptr, lmcache_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    } else {
      warp_copy<ScalarType>(lmcache_ptr, engine_ptr,
                            shape_desc.scalars_per_head<ScalarType>());
    }
  }
}

template <typename ScalarType, bool lmcache_to_engine, GPUKVFormat format>
__global__ void multi_layer_block_transfer_kernel(
    MemoryObj4<ScalarType> lmcache_objects,
    ScalarType** __restrict__ paged_buffer_ptrs,
    const int64_t* engine_block_ids,
    const int num_blocks_per_object,  // e.g. 16 for lmcache chunk size =
                                      // 256 and block size = 16
    const PageBufferShapeDesc shape_desc,
    const int lmcache_chunk_size,  // e.g., 256, used to calculate global offset
                                   // in LMCache object
    const int skip_prefix_n_blocks) {
  // blockIdx.y spans all blocks across all objects (total_blocks).
  // Derive which object and local block index from the flat index.
  const int flat_block_idx = blockIdx.y;
  if (flat_block_idx < skip_prefix_n_blocks) {
    return;
  }
  const int obj_idx = flat_block_idx / num_blocks_per_object;
  const int block_idx_in_object = flat_block_idx % num_blocks_per_object;

  const int engine_block_idx = engine_block_ids[flat_block_idx];
  multi_layer_block_transfer_single_block<ScalarType, lmcache_to_engine,
                                          format>(
      lmcache_objects.objects[obj_idx], paged_buffer_ptrs, engine_block_idx,
      block_idx_in_object * shape_desc.bs,  // offset in LMCache object
      shape_desc, lmcache_chunk_size);
}

// ---------------------------------------------------------------------------
// Pinned ring buffer: pre-allocated cudaHostAlloc region for staging small
// host data (e.g. block_ids) that the GPU kernel reads via zero-copy.
//
// Thread-local so no locking needed across threads. Pinned host memory is
// accessible from any GPU device, so one buffer per thread is sufficient
// regardless of which device is active.
//
// Multi-stream: tracks which streams have outstanding borrows. On wrap,
// syncs all of them to guarantee every borrowed region has been consumed
// before reuse. Wrap is rare (32 MB / ~512 B per call ≈ 65 K calls).
//
// After each kernel launch, the caller registers a cudaLaunchHostFunc
// callback to release the borrowed region back to the ring buffer.
// ---------------------------------------------------------------------------

static constexpr size_t RING_BUFFER_CAPACITY = 32 * 1024 * 1024;  // 32 MB
static constexpr size_t RING_BUFFER_ALIGNMENT = 256;

class PinnedRingBuffer {
 public:
  PinnedRingBuffer() : buffer_(nullptr), head_(0), tail_(0) {
    C10_CUDA_CHECK(
        cudaHostAlloc(&buffer_, RING_BUFFER_CAPACITY, cudaHostAllocMapped));
  }

  ~PinnedRingBuffer() {
    if (buffer_) {
      // At thread exit the CUDA context may already be torn down.
      // Ignore errors — the OS reclaims the memory on process exit.
      cudaFreeHost(buffer_);
    }
  }

  PinnedRingBuffer(const PinnedRingBuffer&) = delete;
  PinnedRingBuffer& operator=(const PinnedRingBuffer&) = delete;

  // Borrow a region of `size` bytes. Returns a pinned host pointer that is
  // accessible from both CPU and GPU (via CUDA unified addressing).
  void* borrow(size_t size, cudaStream_t stream) {
    size_t aligned = align_up(size);
    if (head_ + aligned > RING_BUFFER_CAPACITY) {
      // Wrap: sync ALL streams that have outstanding borrows to ensure
      // every borrowed region has been consumed before we reuse the buffer.
      for (cudaStream_t s : active_streams_) {
        C10_CUDA_CHECK(cudaStreamSynchronize(s));
      }
      active_streams_.clear();
      head_ = 0;
      tail_ = 0;
    }
    void* ptr = buffer_ + head_;
    head_ += aligned;
    active_streams_.insert(stream);
    return ptr;
  }

  // Called asynchronously from cudaLaunchHostFunc after the kernel that
  // uses the borrowed region has completed on the stream.
  void release(size_t size) { tail_ += align_up(size); }

 private:
  char* buffer_;
  size_t head_;               // next byte to allocate
  std::atomic<size_t> tail_;  // advanced by async callbacks
  std::unordered_set<cudaStream_t> active_streams_;

  static size_t align_up(size_t size) {
    return (size + RING_BUFFER_ALIGNMENT - 1) & ~(RING_BUFFER_ALIGNMENT - 1);
  }
};

PinnedRingBuffer& get_ring_buffer() {
  thread_local PinnedRingBuffer ring;
  return ring;
}

struct RingReleaseInfo {
  PinnedRingBuffer* ring;
  size_t size;
};

void CUDART_CB ring_release_callback(void* data) {
  auto* info = static_cast<RingReleaseInfo*>(data);
  info->ring->release(info->size);
  delete info;
}

#define LAUNCH_BLOCK_KERNEL_WITH_FORMAT(DIRECTION, FORMAT)               \
  multi_layer_block_transfer_kernel<uint4, DIRECTION, FORMAT>            \
      <<<grid, block, 0, stream>>>(lmcache_obj4, paged_buffer_ptrs,      \
                                   block_ids_ptr, num_blocks_per_object, \
                                   shape_desc, lmcache_chunk_size,       \
                                   skip_prefix_n_blocks);                \
  C10_CUDA_KERNEL_LAUNCH_CHECK();

}  // namespace

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, std::vector<int64_t> block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    GPUKVFormat gpu_kv_format, int skip_prefix_n_blocks) {
  // --- Validation ---
  int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);

  int total_blocks = static_cast<int>(block_ids.size());
  TORCH_CHECK(total_blocks % num_objects == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_objects (", num_objects,
              ")");
  int num_blocks_per_object = total_blocks / num_objects;

  TORCH_CHECK(num_blocks_per_object * shape_desc.bs == lmcache_chunk_size,
              "blocks_per_object * block_size (",
              num_blocks_per_object * shape_desc.bs,
              ") must equal lmcache_chunk_size (", lmcache_chunk_size, ")");

  TORCH_CHECK(
      shape_desc.hs * shape_desc.element_size % sizeof(uint4) == 0,
      "head_size * element_size (", shape_desc.hs * shape_desc.element_size,
      ") must be divisible by ", sizeof(uint4), " for uint4 vectorized access");

  // --- Build MemoryObj4 ---
  MemoryObj4<uint4> lmcache_obj4;
  lmcache_obj4.num_objects = num_objects;
  for (int i = 0; i < 4; ++i) {
    lmcache_obj4.objects[i] =
        (i < num_objects) ? reinterpret_cast<uint4*>(lmcache_objects_ptrs[i])
                          : nullptr;
  }

  // --- Build paged buffer pointer array ---
  uint4** paged_buffer_ptrs =
      reinterpret_cast<uint4**>(paged_buffer_ptrs_tensor.data_ptr());

  const at::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // --- Stage block_ids into pinned ring buffer (GPU-accessible via UVA) ---
  size_t block_ids_bytes = total_blocks * sizeof(int64_t);
  PinnedRingBuffer& ring = get_ring_buffer();
  int64_t* block_ids_pinned =
      static_cast<int64_t*>(ring.borrow(block_ids_bytes, stream));
  std::memcpy(block_ids_pinned, block_ids.data(), block_ids_bytes);
  const int64_t* block_ids_ptr = block_ids_pinned;

  // --- Grid and block dimensions ---
  int elements_per_head =
      shape_desc.hs * shape_desc.element_size / sizeof(uint4);
  int thread_dim_x = std::min(elements_per_head, 32);
  int thread_dim_y = shape_desc.nh;

  dim3 block(thread_dim_x, thread_dim_y);
  dim3 grid(shape_desc.kv_size, total_blocks, shape_desc.nl);

  // --- Dispatch on direction x format ---
  if (direction == TransferDirection::H2D) {
    switch (gpu_kv_format) {
      case GPUKVFormat::NB_NL_TWO_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true, GPUKVFormat::NB_NL_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true,
                                        GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true,
                                        GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_BS_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true, GPUKVFormat::NL_X_NB_BS_HS);
        break;
      case GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true,
                                        GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NBBS_ONE_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(true, GPUKVFormat::NL_X_NBBS_ONE_HS);
        break;
      default:
        TORCH_CHECK(false, "Unsupported GPUKVFormat: ",
                    static_cast<int>(gpu_kv_format));
    }
  } else {
    switch (gpu_kv_format) {
      case GPUKVFormat::NB_NL_TWO_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false, GPUKVFormat::NB_NL_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false,
                                        GPUKVFormat::NL_X_TWO_NB_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false,
                                        GPUKVFormat::NL_X_NB_TWO_BS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NB_BS_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false, GPUKVFormat::NL_X_NB_BS_HS);
        break;
      case GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false,
                                        GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS);
        break;
      case GPUKVFormat::NL_X_NBBS_ONE_HS:
        LAUNCH_BLOCK_KERNEL_WITH_FORMAT(false, GPUKVFormat::NL_X_NBBS_ONE_HS);
        break;
      default:
        TORCH_CHECK(false, "Unsupported GPUKVFormat: ",
                    static_cast<int>(gpu_kv_format));
    }
  }

  // --- Release ring buffer region after kernel completes on stream ---
  auto* release_info = new RingReleaseInfo{&ring, block_ids_bytes};
  C10_CUDA_CHECK(
      cudaLaunchHostFunc(stream, ring_release_callback, release_info));
}
