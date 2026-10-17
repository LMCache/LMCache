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

using namespace lmcache_mp;

template <typename ScalarType>
void multi_layer_block_kv_transfer_templated(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
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
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
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

namespace {

// Walks an object-group plan: staging copies, bounds checks, ordering and
// phase timing. Each kernel launch is delegated to
// launch_one(group, launch, block_ids_addr, stream).
template <typename LaunchOne>
void run_object_group_plan(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps, bool phase_timing_enabled,
    const std::string& session_id, const LaunchOne& launch_one) {
  // Set the device guard once for the whole plan so every staging copy and
  // kernel launch below is enqueued on this device's current stream, in order.
  const at::cuda::OptionalCUDAGuard device_guard(device);
  // Hoisted: evaluated per section otherwise, even with timing off.
  const cudaStream_t transfer_stream = at::cuda::getCurrentCUDAStream();
  const bool is_h2d = (direction == TransferDirection::H2D);

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

        const uintptr_t block_ids_addr =
            group.block_ids_base +
            static_cast<uintptr_t>(launch.block_ids_offset) * sizeof(int64_t);
        launch_one(group, launch, block_ids_addr, transfer_stream);
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

}  // namespace

void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps, bool phase_timing_enabled,
    const std::string& session_id) {
  const auto int64_opts = at::TensorOptions().dtype(at::kLong).device(device);
  run_object_group_plan(
      direction, device, host_buffer_alignment, kernel_group_specs, batch_steps,
      phase_timing_enabled, session_id,
      [&](const KernelGroupSpec& group, const LaunchVar& launch,
          uintptr_t block_ids_addr, cudaStream_t /*stream*/) {
        // Wrap the plan's pre-resolved raw device addresses as non-owning
        // tensor views so we can reuse the existing
        // multi_layer_block_kv_transfer entry point without touching any of its
        // code. The backing storage is owned by the caller's tensors (kept
        // alive for the duration of this call); these views only carry the
        // pointer/shape each launch needs. Downstream only reads
        // paged_buffer_ptrs_tensor.data_ptr() and block_ids.{data_ptr,
        // size(0)}.
        const at::Tensor paged_buffer_ptrs_tensor = at::from_blob(
            reinterpret_cast<void*>(group.paged_buffer_ptrs), {1}, int64_opts);
        const at::Tensor block_ids = at::from_blob(
            reinterpret_cast<void*>(block_ids_addr),
            {static_cast<int64_t>(launch.total_blocks)}, int64_opts);
        std::vector<int64_t> lmcache_objects_ptrs(
            group.lmcache_objects_ptrs.begin(),
            group.lmcache_objects_ptrs.begin() + launch.num_objects);

        multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor, std::move(lmcache_objects_ptrs),
            block_ids, device, direction, group.shape_desc,
            group.lmcache_chunk_size, group.engine_kv_format,
            launch.skip_prefix_n_blocks);
      });
}

void execute_object_group_transfer_ext(
    TransferDirection direction, int device_index, size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps, lmcache_ext::LaunchFn launch_fn,
    void* user_data, bool phase_timing_enabled, const char* session_id) {
  TORCH_CHECK(launch_fn != nullptr, "launch_fn must not be null");
  run_object_group_plan(
      direction, torch::Device(torch::kCUDA, device_index),
      host_buffer_alignment, kernel_group_specs, batch_steps,
      phase_timing_enabled, session_id ? session_id : "",
      [&](const KernelGroupSpec& group, const LaunchVar& launch,
          uintptr_t block_ids_addr, cudaStream_t stream) {
        launch_fn(user_data, group, launch,
                  reinterpret_cast<const int64_t*>(block_ids_addr), stream);
      });
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

void execute_direct_copy_transfer(
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
#if LMC_HAS_BATCH_MEMCPY
  const bool is_h2d = (direction == TransferDirection::H2D);

  // --- Per-group addressing, resolved once ---
  std::vector<BlockAddressing> addressing(group_specs.size());
  std::vector<int> blocks_per_chunk(group_specs.size());
  std::vector<size_t> layer_bytes(group_specs.size());
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
    layer_bytes[g] = static_cast<size_t>(spec.slots_per_chunk) * sd.nh * sd.hs *
                     sd.element_size;
    const size_t ptrs_needed =
        addressing[g].base == BlockAddressing::Base::kSingleTensor ? 1
        : addressing[g].base == BlockAddressing::Base::kPerPlaneLayer
            ? static_cast<size_t>(sd.kv_size) * sd.nl
            : static_cast<size_t>(sd.nl);
    TORCH_CHECK(spec.paged_layer_ptrs.size() >= ptrs_needed, "group ", g,
                " has ", spec.paged_layer_ptrs.size(),
                " paged layer pointers, needs ", ptrs_needed);
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
  // call consumes the host arrays before it returns.
  std::vector<void*> dsts;
  std::vector<void*> srcs;
  std::vector<size_t> sizes;

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
          static_cast<size_t>(sd.kv_size) * sd.nl * layer_bytes[g];
      TORCH_CHECK(region_end <= obj.nbytes, "group ", g,
                  " region ends at byte ", region_end, " past the object size ",
                  obj.nbytes);

      for (int kv = 0; kv < sd.kv_size; ++kv) {
        for (int layer = 0; layer < sd.nl; ++layer) {
          const uintptr_t dev_base = select_base(addr, spec, kv, layer) +
                                     kv * addr.kv_stride +
                                     layer * addr.layer_stride;
          const uintptr_t host_base =
              obj.host_ptr + spec.byte_offset_in_object +
              (static_cast<size_t>(kv) * sd.nl + layer) * layer_bytes[g];
          for (int b = skip; b < bpc; ++b) {
            const int64_t block_id = spec.block_ids[block_base + b];
            TORCH_CHECK(block_id >= 0 && block_id < sd.nb, "block id ",
                        block_id, " out of range [0, ", sd.nb, ") for group ",
                        g);
            const uintptr_t host = host_base + b * addr.block_bytes;
            const uintptr_t dev =
                dev_base + static_cast<size_t>(block_id) * addr.block_stride;
            append_split(dsts, srcs, sizes, host, dev, addr.block_bytes,
                         obj.host_offset + (host - obj.host_ptr),
                         host_buffer_alignment, is_h2d);
          }
        }
      }
    }
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
  }
#endif
}
