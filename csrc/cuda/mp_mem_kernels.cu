// SPDX-License-Identifier: Apache-2.0

#include "mp_mem_kernels.cuh"
#include "mp_mem_kernels_detail.cuh"

#include "phase_timing_recorder.cuh"

#include <cstdlib>
#include <deque>
#include <mutex>

#if defined(USE_ROCM)
  #include <hip/hip_version.h>
#endif

namespace {

using lmcache_mp::dispatch_by_transfer_unit;
using lmcache_mp::launch_multi_layer_block_transfer;
using lmcache_mp::MemoryObj4;

template <typename ScalarType>
void multi_layer_block_kv_transfer_templated(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<uintptr_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  // --- Validation ---
  int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);

  // --- Build MemoryObj4 ---
  MemoryObj4<ScalarType> lmcache_obj4;
  lmcache_obj4.num_objects = num_objects;
  for (int i = 0; i < 4; ++i) {
    lmcache_obj4.objects[i] =
        (i < num_objects)
            ? reinterpret_cast<ScalarType*>(lmcache_objects_ptrs[i])
            : nullptr;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#if defined(USE_ROCM)
  // hipHostRegister'd memory can be mapped at a different device address
  // (hipDeviceAttributeCanUseHostPointerForRegisteredMem == 0 on MI300/MI355).
  for (int i = 0; i < num_objects; ++i) {
    hipPointerAttribute_t attrs;
    const hipError_t err =
        hipPointerGetAttributes(&attrs, lmcache_obj4.objects[i]);
    TORCH_CHECK(err == hipSuccess,
                "hipPointerGetAttributes failed: ", hipGetErrorString(err));
    if (attrs.type == hipMemoryTypeHost) {
      TORCH_CHECK(attrs.devicePointer != nullptr,
                  "host LMCache object is not mapped on the device");
      lmcache_obj4.objects[i] = static_cast<ScalarType*>(attrs.devicePointer);
    }
  }
#endif

  launch_multi_layer_block_transfer<ScalarType>(
      lmcache_obj4, num_objects,
      reinterpret_cast<ScalarType**>(paged_buffer_ptrs_tensor.data_ptr()),
      block_ids.data_ptr<int64_t>(), static_cast<int>(block_ids.size(0)),
      direction, shape_desc, lmcache_chunk_size, engine_kv_format,
      skip_prefix_n_blocks, stream);
}

/**
 * Calculate the bytes the staging sections of one step move: the sum of
 * its staging copies, i.e. whole memory objects.
 */
int64_t calculate_staging_section_bytes(const BatchStep& step) {
  int64_t bytes = 0;
  for (const auto& copy : step.staging) {
    bytes += static_cast<int64_t>(copy.nbytes);
  }
  return bytes;
}

/**
 * Calculate the bytes the kernel section of one step moves. Derived from
 * the launches rather than the staged payload, because a launch may skip
 * leading blocks (skip_prefix_n_blocks) that the staging copies still
 * carried. Launches with an out-of-range group_idx contribute nothing;
 * the executor's launch loop rejects such a plan via TORCH_CHECK.
 */
int64_t calculate_kernel_section_bytes(
    const BatchStep& step,
    const std::vector<KernelGroupSpec>& kernel_group_specs) {
  int64_t bytes = 0;
  for (const auto& launch : step.launches) {
    if (launch.group_idx < 0 ||
        launch.group_idx >= static_cast<int>(kernel_group_specs.size())) {
      continue;
    }
    const PageBufferShapeDesc& desc =
        kernel_group_specs[launch.group_idx].shape_desc;
    const int64_t block_bytes = static_cast<int64_t>(desc.kv_size) * desc.nl *
                                desc.bs * desc.nh * desc.hs * desc.element_size;
    const int64_t moved_blocks =
        static_cast<int64_t>(launch.total_blocks) - launch.skip_prefix_n_blocks;
    if (moved_blocks > 0) {
      bytes += block_bytes * moved_blocks;
    }
  }
  return bytes;
}

}  // namespace

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<uintptr_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  dispatch_by_transfer_unit(
      shape_desc, engine_kv_format, [&](auto transfer_unit) {
        using ScalarType = typename decltype(transfer_unit)::type;
        multi_layer_block_kv_transfer_templated<ScalarType>(
            paged_buffer_ptrs_tensor, lmcache_objects_ptrs, block_ids, device,
            direction, shape_desc, lmcache_chunk_size, engine_kv_format,
            skip_prefix_n_blocks);
      });
}

void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps, bool phase_timing_enabled,
    const std::string& session_id) {
  // Set the device guard once for the whole plan so every staging copy and
  // kernel launch below is enqueued on this device's current stream, in order.
  const at::cuda::OptionalCUDAGuard device_guard(device);
  // Hoisted: evaluated per section otherwise, even with timing off.
  const cudaStream_t transfer_stream = at::cuda::getCurrentCUDAStream();
  const bool is_h2d = (direction == TransferDirection::H2D);
  const auto int64_opts = at::TensorOptions().dtype(at::kLong).device(device);

  const auto do_staging = [&](const std::vector<StagingCopy>& staging) {
    for (const auto& copy : staging) {
      lmcache_memcpy_async(copy.dest, copy.src, copy.nbytes, direction,
                           copy.host_offset, host_buffer_alignment);
    }
  };

  PhaseTimer phase_timer(
      phase_timing_enabled, transfer_stream, static_cast<int>(direction),
      static_cast<int>(device.index()), batch_steps.size() * 2, session_id);

  for (const auto& step : batch_steps) {
    const int64_t step_bytes = calculate_staging_section_bytes(step);
    const int64_t kernel_bytes =
        calculate_kernel_section_bytes(step, kernel_group_specs);

    // H2D stages CPU->GPU temp buffers before the kernel reads them; D2H stages
    // GPU->CPU after the kernel writes them. The per-step ordering must be
    // preserved because temp buffers are reused across steps.
    if (is_h2d) {
      auto section = phase_timer.section(TransferPhase::STAGING, step_bytes,
                                         transfer_stream);
      do_staging(step.staging);
    }
    {
      auto section = phase_timer.section(TransferPhase::KERNEL, kernel_bytes,
                                         transfer_stream);
      for (const auto& launch : step.launches) {
        TORCH_CHECK(
            launch.group_idx >= 0 &&
                launch.group_idx < static_cast<int>(kernel_group_specs.size()),
            "LaunchVar.group_idx out of range: ", launch.group_idx);
        const KernelGroupSpec& group = kernel_group_specs[launch.group_idx];
        TORCH_CHECK(launch.num_objects >= 1 &&
                        launch.num_objects <=
                            static_cast<int>(group.lmcache_objects_ptrs.size()),
                    "LaunchVar.num_objects (", launch.num_objects,
                    ") exceeds available temp buffers (",
                    group.lmcache_objects_ptrs.size(), ")");
        // Bounds-check the block_ids slice before the kernel dereferences it on
        // device: an out-of-range offset/length would otherwise be a silent
        // out-of-bounds device read (CUDA fault or garbage), not a clean error.
        TORCH_CHECK(launch.block_ids_offset >= 0,
                    "LaunchVar.block_ids_offset must be non-negative, got ",
                    launch.block_ids_offset);
        TORCH_CHECK(launch.total_blocks >= 0,
                    "LaunchVar.total_blocks must be non-negative, got ",
                    launch.total_blocks);
        TORCH_CHECK(launch.block_ids_offset + launch.total_blocks <=
                        group.block_ids_capacity,
                    "LaunchVar block_ids slice [", launch.block_ids_offset,
                    ", ", launch.block_ids_offset + launch.total_blocks,
                    ") exceeds block_ids capacity ", group.block_ids_capacity);

        // Wrap the plan's pre-resolved raw device addresses as non-owning
        // tensor views so we can reuse the existing
        // multi_layer_block_kv_transfer entry point without touching any of its
        // code. The backing storage is owned by the caller's tensors (kept
        // alive for the duration of this call); these views only carry the
        // pointer/shape each launch needs. Downstream only reads
        // paged_buffer_ptrs_tensor.data_ptr() and block_ids.{data_ptr,
        // size(0)}.
        const uintptr_t block_ids_addr =
            group.block_ids_base +
            static_cast<uintptr_t>(launch.block_ids_offset) * sizeof(int64_t);
        const at::Tensor paged_buffer_ptrs_tensor = at::from_blob(
            reinterpret_cast<void*>(group.paged_buffer_ptrs), {1}, int64_opts);
        const at::Tensor block_ids = at::from_blob(
            reinterpret_cast<void*>(block_ids_addr),
            {static_cast<int64_t>(launch.total_blocks)}, int64_opts);
        std::vector<uintptr_t> lmcache_objects_ptrs(
            group.lmcache_objects_ptrs.begin(),
            group.lmcache_objects_ptrs.begin() + launch.num_objects);

        multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor, std::move(lmcache_objects_ptrs),
            block_ids, device, direction, group.shape_desc,
            group.lmcache_chunk_size, group.engine_kv_format,
            launch.skip_prefix_n_blocks);
      }
    }
    if (!is_h2d) {
      auto section = phase_timer.section(TransferPhase::STAGING, step_bytes,
                                         transfer_stream);
      do_staging(step.staging);
    }
  }

  // Publish only after the whole plan enqueued successfully.
  phase_timer.commit();
}

// ---------------------------------------------------------------------------
// Direct copy-engine transfer (cudaMemcpyBatchAsync)
// ---------------------------------------------------------------------------

// cudaMemcpyBatchAsync exists from CUDA 12.8. HIP gained hipMemcpyBatchAsync
// in ROCm 7.x (HIP 7.15) with the identical 9-arg (failIdx) signature, so the
// same direct copy-engine path serves both once the sources are hipified.
#if !defined(USE_ROCM) && defined(CUDART_VERSION) && CUDART_VERSION >= 12080
  #define LMC_HAS_BATCH_MEMCPY 1
#elif defined(USE_ROCM) && defined(HIP_VERSION) && HIP_VERSION >= 71500000
  #define LMC_HAS_BATCH_MEMCPY 1
#else
  #define LMC_HAS_BATCH_MEMCPY 0
#endif

// The 2-D path (cudaMemcpy3DBatchAsync) is CUDA-only; HIP builds keep one
// 1-D entry per layer.
#if LMC_HAS_BATCH_MEMCPY && !defined(USE_ROCM)
  #define LMC_HAS_2D_BATCH_MEMCPY 1
#else
  #define LMC_HAS_2D_BATCH_MEMCPY 0
#endif

namespace {

// Byte-level addressing of one paged block, resolved once per kernel group.
// Every eligible format is affine in (kv plane, layer, block):
//   address = base(kv, layer) + kv * kv_stride + layer * layer_stride
//             + block_id * block_stride
// The strides mirror calculate_engine_global_offset() above (times the
// element size), so `scalars_per_block` -- and with it the physical
// block_stride_elems padding -- is honoured identically on both paths.
struct BlockAddressing {
  enum class Base { kPerLayer, kSingleTensor, kPerPlaneLayer };
  Base base;
  size_t kv_stride;     // bytes between the K and V planes of one block
  size_t layer_stride;  // bytes between layers (cross-layer formats only)
  size_t block_stride;  // bytes between consecutive block ids
  size_t block_bytes;   // tight bs * nh * hs * element_size moved per entry
};

bool resolve_block_addressing(EngineKVFormat format,
                              const PageBufferShapeDesc& sd,
                              BlockAddressing& out) {
  const size_t elem = static_cast<size_t>(sd.element_size);
  const size_t tight = static_cast<size_t>(sd.bs) * sd.nh * sd.hs * elem;
  const size_t blk = sd.block_stride_elems > 0
                         ? static_cast<size_t>(sd.block_stride_elems) * elem
                         : tight;
  const size_t kv = static_cast<size_t>(sd.kv_size);
  using B = BlockAddressing::Base;
  switch (format) {
    case EngineKVFormat::NB_NL_TWO_BS_NH_HS:
      // Cross-layer: single tensor [NB, NL, 2, BS, NH, HS]
      out = {B::kSingleTensor, blk, kv * blk, kv * blk * sd.nl, tight};
      return true;
    case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:
      // L tensors [2, NB, BS, NH, HS]
      out = {B::kPerLayer, static_cast<size_t>(sd.nb) * blk, 0, blk, tight};
      return true;
    case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:
      // L tensors [NB, 2, BS, NH, HS]
      out = {B::kPerLayer, blk, 0, kv * blk, tight};
      return true;
    case EngineKVFormat::NL_X_NB_BS_HS:         // MLA [NB, BS, HS]
    case EngineKVFormat::NL_X_NBBS_ONE_HS:      // SGLang MLA [NBBS, 1, HS]
    case EngineKVFormat::NL_X_NB_BS_NH_TWO_HS:  // fused NHD [NB, BS, NH, 2HS]
    case EngineKVFormat::NL_X_NB_BS_NH_CS:      // fused NHD [NB, BS, NH, CS]
    case EngineKVFormat::NL_X_NB_BS_NH_HS:  // SGLang component [NB, BS, NH, HS]
      out = {B::kPerLayer, 0, 0, blk, tight};
      return true;
    case EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS:  // SGLang MHA, K then V ptrs
    case EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS:
      out = {B::kPerPlaneLayer, 0, 0, blk, tight};
      return true;
    default:
      // HND (heads before tokens), blocked-scale and (K, V)-tuple layouts:
      // a paged block is not one contiguous [bs, nh*hs] run.
      return false;
  }
}

inline uintptr_t select_base(const BlockAddressing& addr,
                             const DirectCopyGroupSpec& spec, int kv,
                             int layer) {
  const auto& ptrs = spec.paged_layer_ptrs;
  const int nl = spec.shape_desc.nl;
  switch (addr.base) {
    case BlockAddressing::Base::kSingleTensor:
      return ptrs[0];
    case BlockAddressing::Base::kPerPlaneLayer:
      return ptrs[static_cast<size_t>(kv) * nl + layer];
    case BlockAddressing::Base::kPerLayer:
    default:
      return ptrs[layer];
  }
}

// Append one host<->device range, split so that no piece crosses a
// `host_buffer_alignment` boundary of the allocator's virtual offset (each
// pin chunk is a separate cudaHostRegister region; a copy spanning two of
// them fails with cudaErrorInvalidValue). Same arithmetic as
// lmcache_memcpy_async.
inline void append_split(std::vector<void*>& dsts, std::vector<void*>& srcs,
                         std::vector<size_t>& sizes, uintptr_t host,
                         uintptr_t dev, size_t nbytes,
                         size_t host_virtual_offset,
                         size_t host_buffer_alignment, bool is_h2d) {
  const size_t mask = host_buffer_alignment - 1;
  size_t offset = 0;
  while (offset < nbytes) {
    const size_t aligned_area_end =
        ((offset + host_virtual_offset) & ~mask) + host_buffer_alignment;
    const size_t real_end =
        std::min<size_t>(host_virtual_offset + nbytes, aligned_area_end);
    const size_t piece = real_end - offset - host_virtual_offset;
    void* h = reinterpret_cast<void*>(host + offset);
    void* d = reinterpret_cast<void*>(dev + offset);
    dsts.push_back(is_h2d ? d : h);
    srcs.push_back(is_h2d ? h : d);
    sizes.push_back(piece);
    offset += piece;
  }
}

#if LMC_HAS_2D_BATCH_MEMCPY
// This function calculates the engine-side pitch of a 2-D copy that moves one
// block of every layer of a kernel group at once (width = block bytes,
// height = layers): the byte distance in the engine KV cache from a block of
// layer l to the same block (and kv plane) of layer l + 1.
// Returns:
// - that distance, if it is the same for every pair of consecutive layers and
//   one block of every layer fits inside one block stride. This is the
//   block-outermost layout (e.g. vLLM's BLNHC for DeepSeek-V4), where each
//   block packs one padded page per layer side by side, so the distance is
//   the padded page size;
// - 0 otherwise (e.g. layer-outermost layouts, whose layers are a whole
//   layer apart); the caller then copies each layer with its own 1-D entry.
// A non-zero pitch puts each layer's base inside the previous layer's first
// block, so all layers lie in one allocation, as the rows of a 2-D copy must,
// and the pitch stays far below cudaDevAttrMaxPitch.
size_t uniform_layer_pitch(const BlockAddressing& addr,
                           const DirectCopyGroupSpec& spec) {
  const PageBufferShapeDesc& sd = spec.shape_desc;
  if (sd.nl < 2 || sd.nb < 2) return 0;
  const auto& ptrs = spec.paged_layer_ptrs;
  size_t pitch = addr.layer_stride;
  if (addr.base != BlockAddressing::Base::kSingleTensor) {
    if (ptrs[1] <= ptrs[0]) return 0;
    pitch = ptrs[1] - ptrs[0];
    const int planes =
        addr.base == BlockAddressing::Base::kPerPlaneLayer ? sd.kv_size : 1;
    for (int p = 0; p < planes; ++p) {
      for (int l = 1; l < sd.nl; ++l) {
        const size_t i = static_cast<size_t>(p) * sd.nl + l;
        if (ptrs[i] - ptrs[i - 1] != pitch) return 0;
      }
    }
  }
  const bool fits_in_block =
      pitch >= addr.block_bytes &&
      (sd.nl - 1) * pitch + addr.block_bytes <= addr.block_stride;
  return fits_in_block ? pitch : 0;
}

// Append one 2-D host<->device copy of `height` rows of `width` bytes (row r
// at host + r * host_pitch and dev + r * dev_pitch), split like append_split
// so no op crosses a `host_buffer_alignment` boundary of the allocator's
// virtual offset: consecutive rows inside one pin chunk stay one op, and a row
// that straddles a boundary becomes single-row pieces.
void append_2d_split(std::vector<cudaMemcpy3DBatchOp>& ops, uintptr_t host,
                     uintptr_t dev, size_t width, size_t height,
                     size_t host_pitch, size_t dev_pitch,
                     size_t host_virtual_offset, size_t host_buffer_alignment,
                     bool is_h2d) {
  auto emit = [&](uintptr_t h, uintptr_t d, size_t w, size_t rows) {
    cudaMemcpy3DBatchOp op{};
    cudaMemcpy3DOperand host_op{};
    host_op.type = cudaMemcpyOperandTypePointer;
    host_op.op.ptr.ptr = reinterpret_cast<void*>(h);
    host_op.op.ptr.rowLength = host_pitch;
    host_op.op.ptr.layerHeight = rows;
    cudaMemcpy3DOperand dev_op{};
    dev_op.type = cudaMemcpyOperandTypePointer;
    dev_op.op.ptr.ptr = reinterpret_cast<void*>(d);
    dev_op.op.ptr.rowLength = dev_pitch;
    dev_op.op.ptr.layerHeight = rows;
    op.src = is_h2d ? host_op : dev_op;
    op.dst = is_h2d ? dev_op : host_op;
    op.extent = make_cudaExtent(w, rows, 1);  // bytes, rows, 1 layer
    op.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    ops.push_back(op);
  };
  // Consecutive rows inside one pin chunk accumulate into a pending 2-D op:
  // `run_start` is the first row of the pending op, `run_rows` is how many
  // rows it has (0 = nothing pending), and `run_chunk` is the index of the
  // cudaHostRegister chunk (virtual offset / host_buffer_alignment) that all
  // of its rows lie in. flush() emits the pending op.
  size_t run_start = 0;
  size_t run_rows = 0;
  size_t run_chunk = 0;
  auto flush = [&]() {
    if (run_rows > 0) {
      emit(host + run_start * host_pitch, dev + run_start * dev_pitch, width,
           run_rows);
      run_rows = 0;
    }
  };
  for (size_t r = 0; r < height; ++r) {
    // Pin chunks holding the first and the last byte of row r.
    const size_t v = host_virtual_offset + r * host_pitch;
    const size_t first_chunk = v / host_buffer_alignment;
    const size_t last_chunk = (v + width - 1) / host_buffer_alignment;
    if (first_chunk != last_chunk) {
      flush();
      size_t off = 0;
      while (off < width) {
        const size_t boundary =
            ((v + off) / host_buffer_alignment + 1) * host_buffer_alignment;
        const size_t end = std::min(width, boundary - v);
        emit(host + r * host_pitch + off, dev + r * dev_pitch + off, end - off,
             1);
        off = end;
      }
      continue;
    }
    if (run_rows > 0 && first_chunk == run_chunk) {
      ++run_rows;
    } else {
      flush();
      run_start = r;
      run_rows = 1;
      run_chunk = first_chunk;
    }
  }
  flush();
}
#endif

}  // namespace

bool batch_memcpy_supported() {
#if LMC_HAS_BATCH_MEMCPY
  static const bool supported = [] {
    int runtime = 0;
    int driver = 0;
    if (cudaRuntimeGetVersion(&runtime) != cudaSuccess) return false;
    if (cudaDriverGetVersion(&driver) != cudaSuccess) return false;
  #if defined(USE_ROCM)
    // HIP versions its runtime/driver on a different scale, so the compile-time
    // HIP_VERSION guard cannot by itself prove the op is functional: some HIP
    // builds link hipMemcpyBatchAsync but return hipErrorNotSupported at
    // runtime. Keep the ROCm direct copy-engine path opt-in -- it is enabled
    // only when LMCACHE_ROCM_ENABLE_BATCH_MEMCPY is set to a non-zero value.
    (void)runtime;
    (void)driver;
    const char* enable = std::getenv("LMCACHE_ROCM_ENABLE_BATCH_MEMCPY");
    if (enable == nullptr || enable[0] == '\0' || enable[0] == '0') {
      return false;
    }
    // hipMemcpyBatchAsync's copy engine dereferences the host operand directly,
    // so the hipHostRegister'd pinned host pool must be device-addressable.
    // Where hipDeviceAttributeCanUseHostPointerForRegisteredMem == 0 (observed
    // on MI300/MI355 ROCm configs) the registered host VA is not device-usable
    // and the batched D2H/H2D copy faults asynchronously, so require the
    // attribute before selecting the copy-engine path.
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
      return false;
    }
    // Some ROCm builds report 0 here even though the registered host VA is
    // device-addressable; allow an explicit opt-in after the parity tests.
    const char* force = std::getenv("LMCACHE_ROCM_FORCE_BATCH_MEMCPY");
    if (force != nullptr && force[0] != '\0' && force[0] != '0') {
      return true;
    }
    int can_use_host_ptr = 0;
    if (hipDeviceGetAttribute(
            &can_use_host_ptr,
            hipDeviceAttributeCanUseHostPointerForRegisteredMem,
            device) != hipSuccess) {
      return false;
    }
    return can_use_host_ptr != 0;
  #else
    return runtime >= 12080 && driver >= 12080;
  #endif
  }();
  return supported;
#else
  return false;
#endif
}

bool direct_copy_format_supported(EngineKVFormat engine_kv_format) {
  BlockAddressing unused;
  PageBufferShapeDesc probe{};
  probe.kv_size = 1;
  probe.nl = 1;
  probe.nb = 1;
  probe.bs = 1;
  probe.nh = 1;
  probe.hs = 1;
  probe.element_size = 1;
  probe.block_stride_elems = 0;
  return resolve_block_addressing(engine_kv_format, probe, unused);
}

size_t execute_direct_copy_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<DirectCopyGroupSpec>& group_specs,
    const std::vector<DirectCopyObject>& objects) {
  TORCH_CHECK(batch_memcpy_supported(),
              "cudaMemcpyBatchAsync is not available: the extension must be "
              "built against CUDA >= 12.8 and run on a >= 12.8 runtime/driver");
  TORCH_CHECK(host_buffer_alignment > 0 &&
                  (host_buffer_alignment & (host_buffer_alignment - 1)) == 0,
              "host_buffer_alignment must be a power of two, got ",
              host_buffer_alignment);
  size_t num_ops = 0;
#if LMC_HAS_BATCH_MEMCPY
  const bool is_h2d = (direction == TransferDirection::H2D);

  // --- Per-group addressing, resolved once ---
  std::vector<BlockAddressing> addressing(group_specs.size());
  std::vector<int> blocks_per_chunk(group_specs.size());
  // Bytes of one (kv plane, layer) region of a chunk in the LMCache object.
  std::vector<size_t> lmcache_layer_bytes(group_specs.size());
  // Bytes from a block of one layer to the same block of the next layer in
  // the engine KV cache; non-zero means one 2-D copy moves the block of every
  // layer (see uniform_layer_pitch).
  std::vector<size_t> engine_layer_pitch(group_specs.size());
  for (size_t g = 0; g < group_specs.size(); ++g) {
    const DirectCopyGroupSpec& spec = group_specs[g];
    const PageBufferShapeDesc& sd = spec.shape_desc;
    TORCH_CHECK(
        resolve_block_addressing(spec.engine_kv_format, sd, addressing[g]),
        "EngineKVFormat ", static_cast<int>(spec.engine_kv_format),
        " is not eligible for the direct copy path");
    TORCH_CHECK(sd.bs > 0 && spec.slots_per_chunk % sd.bs == 0,
                "slots_per_chunk (", spec.slots_per_chunk,
                ") must be a positive multiple of block size (", sd.bs, ")");
    blocks_per_chunk[g] = spec.slots_per_chunk / sd.bs;
    lmcache_layer_bytes[g] = static_cast<size_t>(spec.slots_per_chunk) * sd.nh *
                             sd.hs * sd.element_size;
    const size_t ptrs_needed =
        addressing[g].base == BlockAddressing::Base::kSingleTensor ? 1
        : addressing[g].base == BlockAddressing::Base::kPerPlaneLayer
            ? static_cast<size_t>(sd.kv_size) * sd.nl
            : static_cast<size_t>(sd.nl);
    TORCH_CHECK(spec.paged_layer_ptrs.size() >= ptrs_needed, "group ", g,
                " has ", spec.paged_layer_ptrs.size(),
                " paged layer pointers, needs ", ptrs_needed);
  #if LMC_HAS_2D_BATCH_MEMCPY
    engine_layer_pitch[g] = uniform_layer_pitch(addressing[g], spec);
  #endif
  }

  const at::cuda::OptionalCUDAGuard device_guard(device);
  // cudaMemcpyBatchAsync rejects the legacy NULL stream. The transfer runs on
  // the cache context's own stream in production; if a caller is on the
  // legacy default stream, use the per-thread default stream, which
  // synchronizes with the legacy stream and so preserves the ordering.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (stream == nullptr) {
    stream = cudaStreamPerThread;
  }

  cudaMemcpyAttributes attrs{};
  attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t attrs_idx = 0;

  // Entry tables are rebuilt per object and reused across objects: the CUDA
  // call consumes the host arrays before it returns. 1-D entries (one per
  // kv plane, layer and block) go to cudaMemcpyBatchAsync; 2-D ops (one per
  // kv plane and block, covering every layer) to cudaMemcpy3DBatchAsync.
  std::vector<void*> dsts;
  std::vector<void*> srcs;
  std::vector<size_t> sizes;
  #if LMC_HAS_2D_BATCH_MEMCPY
  std::vector<cudaMemcpy3DBatchOp> ops;
  #endif

  for (const DirectCopyObject& obj : objects) {
    TORCH_CHECK(obj.skip_prefix_n_blocks.size() == group_specs.size(),
                "DirectCopyObject.skip_prefix_n_blocks has ",
                obj.skip_prefix_n_blocks.size(), " entries, expected ",
                group_specs.size());
    TORCH_CHECK(obj.chunk_idx >= 0, "chunk_idx must be non-negative, got ",
                obj.chunk_idx);
    dsts.clear();
    srcs.clear();
    sizes.clear();
  #if LMC_HAS_2D_BATCH_MEMCPY
    ops.clear();
  #endif

    for (size_t g = 0; g < group_specs.size(); ++g) {
      const DirectCopyGroupSpec& spec = group_specs[g];
      const PageBufferShapeDesc& sd = spec.shape_desc;
      const BlockAddressing& addr = addressing[g];
      const int bpc = blocks_per_chunk[g];
      const int skip = obj.skip_prefix_n_blocks[g];
      TORCH_CHECK(skip >= 0 && skip <= bpc, "skip_prefix_n_blocks (", skip,
                  ") out of range [0, ", bpc, "] for group ", g);
      const size_t block_base = static_cast<size_t>(obj.chunk_idx) * bpc;
      TORCH_CHECK(block_base + bpc <= spec.block_ids.size(), "chunk ",
                  obj.chunk_idx, " needs block ids [", block_base, ", ",
                  block_base + bpc, ") but group ", g, " only has ",
                  spec.block_ids.size());
      // Whole region this group touches inside the object; one check covers
      // every entry below because (kv, layer, block) only ever move forward.
      const size_t region_end =
          spec.byte_offset_in_object +
          static_cast<size_t>(sd.kv_size) * sd.nl * lmcache_layer_bytes[g];
      TORCH_CHECK(region_end <= obj.nbytes, "group ", g,
                  " region ends at byte ", region_end, " past the object size ",
                  obj.nbytes);

      // With a uniform layer pitch, one 2-D copy starting at layer 0 covers
      // every layer of a block, so the layer loop runs once.
      const int layers_per_copy = engine_layer_pitch[g] > 0 ? sd.nl : 1;
      for (int kv = 0; kv < sd.kv_size; ++kv) {
        for (int layer = 0; layer < sd.nl; layer += layers_per_copy) {
          const uintptr_t dev_base = select_base(addr, spec, kv, layer) +
                                     kv * addr.kv_stride +
                                     layer * addr.layer_stride;
          const uintptr_t host_base =
              obj.host_ptr + spec.byte_offset_in_object +
              (static_cast<size_t>(kv) * sd.nl + layer) *
                  lmcache_layer_bytes[g];
          for (int b = skip; b < bpc; ++b) {
            const int64_t block_id = spec.block_ids[block_base + b];
            TORCH_CHECK(block_id >= 0 && block_id < sd.nb, "block id ",
                        block_id, " out of range [0, ", sd.nb, ") for group ",
                        g);
            const uintptr_t host = host_base + b * addr.block_bytes;
            const uintptr_t dev =
                dev_base + static_cast<size_t>(block_id) * addr.block_stride;
  #if LMC_HAS_2D_BATCH_MEMCPY
            if (layers_per_copy > 1) {
              append_2d_split(ops, host, dev, addr.block_bytes, sd.nl,
                              lmcache_layer_bytes[g], engine_layer_pitch[g],
                              obj.host_offset + (host - obj.host_ptr),
                              host_buffer_alignment, is_h2d);
              continue;
            }
  #endif
            append_split(dsts, srcs, sizes, host, dev, addr.block_bytes,
                         obj.host_offset + (host - obj.host_ptr),
                         host_buffer_alignment, is_h2d);
          }
        }
      }
    }
  #if LMC_HAS_2D_BATCH_MEMCPY
    if (!ops.empty()) {
    #if CUDART_VERSION >= 13000
      const cudaError_t err =
          cudaMemcpy3DBatchAsync(ops.size(), ops.data(), 0, stream);
    #else
      size_t fail_idx = 0;
      const cudaError_t err =
          cudaMemcpy3DBatchAsync(ops.size(), ops.data(), &fail_idx, 0, stream);
    #endif
      TORCH_CHECK(err == cudaSuccess, "cudaMemcpy3DBatchAsync failed with ",
                  cudaGetErrorString(err), " (", ops.size(), " 2-D ops, chunk ",
                  obj.chunk_idx, ")");
      num_ops += ops.size();
    }
  #endif
    if (dsts.empty()) continue;

  #if CUDART_VERSION >= 13000
    const cudaError_t err =
        cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(),
                             dsts.size(), &attrs, &attrs_idx, 1, stream);
  #else
    // CUDA 12.8/12.9 take an extra out-parameter for the failing index.
    size_t fail_idx = 0;
    const cudaError_t err = cudaMemcpyBatchAsync(
        dsts.data(), srcs.data(), sizes.data(), dsts.size(), &attrs, &attrs_idx,
        1, &fail_idx, stream);
  #endif
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyBatchAsync failed with ",
                cudaGetErrorString(err), " (", dsts.size(), " entries, chunk ",
                obj.chunk_idx, ")");
    num_ops += dsts.size();
  }
#endif
  return num_ops;
}
