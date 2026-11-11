// SPDX-License-Identifier: Apache-2.0

// C++ host layer for the block-level MP-mode KV transfer (mirrors upstream
// LMCache csrc/mp_mem_kernels.cu).
//
// Design contract:
//   - prepare_group resolves a PageBufferShapeDesc + EngineKVFormat into the
//     flat BlockTransferLayout POD ONCE per group, with every static geometry
//     check. Both native entry points consume only PreparedGroup; the
//     device kernel receives the POD verbatim and never re-interprets the
//     format.
//   - validate_launch checks the per-transfer dynamic variables.
//   - The plan executor validates ALL groups and ALL launches before any
//     staging copy or kernel launch is enqueued: an unsupported group
//     fails with "nothing started", never half-enqueued.
//   - The kernel is a pure byte mover; there is no packed/two-plane special
//     case and no dtype dispatch left in this file.

#include "mp_mem_kernels.h"

#include "kernels/multi_layer/multi_layer_block_mem_kernels.h"
#include "managed_mem.h"
#include "tiling/platform/platform_ascendc.h"

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

// DataCopy granularity on the GM side: segment base addresses and segment
// byte lengths must both be multiples of 32B.
constexpr int64_t kGmAlignBytes = 32;
// Single UB segment budget: the depth-2 queue must hold two segments, so one
// segment may use at most half of the total budget.
constexpr int64_t kUbSegmentBytes =
    kvcache_ops::kBlockTransferUbBytes / kvcache_ops::kBlockTransferQueueDepth;
// DataCopyPad GM-side gap fields are uint32.
constexpr int64_t kCopyGmGapMax =
    static_cast<int64_t>(std::numeric_limits<uint32_t>::max());

int64_t align_up32(int64_t v) { return (v + 31) & ~int64_t(31); }

int64_t checked_mul(int64_t a, int64_t b, const char* what) {
  TORCH_CHECK(a >= 0 && b >= 0, what, " must be non-negative, got ", a, " * ",
              b);
  if (a != 0) {
    TORCH_CHECK(b <= std::numeric_limits<int64_t>::max() / a, what,
                " overflows int64: ", a, " * ", b);
  }
  return a * b;
}

int64_t checked_add(int64_t a, int64_t b, const char* what) {
  TORCH_CHECK(a >= 0 && b >= 0 && b <= std::numeric_limits<int64_t>::max() - a,
              what, " overflows int64: ", a, " + ", b);
  return a + b;
}

// Usable per-AIV UB staging budget, queried from the platform: GetCoreMemSize
// reports the PHYSICAL UB size. Returns 0 on failure -- callers then use
// the kernel's built-in 128KB default.
int64_t query_ub_budget_bytes() {
  auto* platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(aclrtGetSocName());
  if (platform == nullptr) {
    return 0;
  }
  uint64_t ub_size = 0;
  platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  return ub_size == 0 ? 0 : static_cast<int64_t>(ub_size);
}

// ---------------------------------------------------------------------------
// prepare_group: static geometry, resolved once per group.
// ---------------------------------------------------------------------------

struct PreparedGroup {
  kvcache_ops::BlockTransferLayout layout{};
  bool separate_plane = false;  // format 16: (layer, plane, block) work items
  bool one_pointer_per_layer = false;  // format 2: ptrs[layer] + plane offset
  int32_t nl = 0;
  int32_t nb = 0;
  int32_t bs = 0;
  int64_t slots_per_object = 0;
  int64_t ub_bytes = 0;  // queried UB budget; 0 = kernel default
};

PreparedGroup prepare_group(const PageBufferShapeDesc& shape_desc,
                            EngineKVFormat engine_kv_format,
                            int64_t slots_per_object, int64_t ub_bytes) {
  PageBufferShapeDesc sd = shape_desc;
  // Format 2 is one [NB, 2, BS, NH, HS] tensor per layer. The pointer table
  // stays one base pointer per layer; plane addresses are base + offset.
  // Axis 1 is two equal slabs (semantic K/V, or opaque page halves).
  // Its LMC geometry matches format 16.
  const bool fmt2 = engine_kv_format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS;
  const bool fmt13 = engine_kv_format == EngineKVFormat::NL_X_NB_BS_NH_CS;
  const bool separate =
      engine_kv_format == EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS || fmt2;
  const bool fmt17 = engine_kv_format == EngineKVFormat::NL_X_NP_X_NB_BS_ONE_HS;
  TORCH_CHECK(fmt13 || separate || fmt17,
              "LMCache-Ascend block-level MP transfer currently supports "
              "NL_X_NB_TWO_BS_NH_HS (2), NL_X_NB_BS_NH_CS (13), "
              "NL_X_TWO_X_NB_BS_NH_HS (16), and "
              "NL_X_NP_X_NB_BS_ONE_HS (17), "
              "got ",
              static_cast<int>(engine_kv_format));
  TORCH_CHECK(sd.nl > 0 && sd.nb > 0 && sd.bs > 0 && sd.nh > 0 && sd.hs > 0,
              "shape descriptor dims must be positive, got nl=", sd.nl,
              " nb=", sd.nb, " bs=", sd.bs, " nh=", sd.nh, " hs=", sd.hs);
  TORCH_CHECK(
      sd.element_size == 1 || sd.element_size == 2 || sd.element_size == 4,
      "element_size must be 1, 2 or 4, got ", sd.element_size);
  TORCH_CHECK(sd.kv_size == (separate ? 2 : 1), "kv_size must be ",
              (separate ? 2 : 1), " for format ",
              static_cast<int>(engine_kv_format), ", got ", sd.kv_size);

  const int64_t scalar_row =
      checked_mul(checked_mul(sd.nh, sd.hs, "nh * hs"), sd.element_size,
                  "LMC scalar row bytes");
  if (fmt13 && sd.num_planes == 0) {
    // Format 13 is one fused [NB, BS, NH, CS] tensor per layer. Its plane
    // geometry is fully determined by the scalar descriptor fields, so no
    // per-plane tensor introspection is needed during registration.
    sd.num_planes = 1;
    sd.plane_slot_bytes[0] = scalar_row;
    sd.plane_block_stride_bytes[0] =
        sd.block_stride_elems > 0
            ? checked_mul(sd.block_stride_elems, sd.element_size,
                          "format 13 block stride bytes")
            : checked_mul(sd.bs, scalar_row, "format 13 block stride bytes");
  }

  // Format 2 is one [NB, 2, BS, NH, HS] tensor. The scalar descriptor
  // fixes both plane addresses, so fill them here.
  if (fmt2) {
    const int64_t row =
        checked_mul(checked_mul(static_cast<int64_t>(sd.nh), sd.hs, "nh * hs"),
                    sd.element_size, "format 2 row bytes");
    const int64_t half = checked_mul(sd.bs, row, "format 2 half page");
    const int64_t tight_elems = checked_mul(
        int64_t{2},
        checked_mul(sd.bs,
                    checked_mul(static_cast<int64_t>(sd.nh), sd.hs, "nh * hs"),
                    "bs * nh * hs"),
        "format 2 tight block elems");
    const int64_t block_elems =
        sd.block_stride_elems > 0 ? sd.block_stride_elems : tight_elems;
    const int64_t block_bytes = checked_mul(block_elems, sd.element_size,
                                            "format 2 block stride bytes");
    sd.num_planes = 2;
    sd.plane_slot_bytes[0] = sd.plane_slot_bytes[1] = row;
    sd.plane_block_stride_bytes[0] = sd.plane_block_stride_bytes[1] =
        block_bytes;
    sd.plane_engine_base_offset_bytes[0] = 0;
    sd.plane_engine_base_offset_bytes[1] = half;
  }

  // Formats 16/17 must carry explicit plane geometry filled at registration;
  // format 13 derives it above when num_planes == 0.
  TORCH_CHECK(sd.num_planes >= 1 && sd.num_planes <= kvcache_ops::kMaxPlanes,
              "num_planes must be in [1, ", kvcache_ops::kMaxPlanes,
              "] after resolving plane geometry, got ", sd.num_planes);
  if (separate) {
    TORCH_CHECK(
        sd.num_planes == 2,
        "separate-plane formats require exactly 2 physical planes, got ",
        sd.num_planes);
  }
  if (fmt13) {
    TORCH_CHECK(sd.num_planes == 1,
                "format 13 requires exactly 1 physical plane, got ",
                sd.num_planes);
  }
  if (fmt17) {
    TORCH_CHECK(sd.nh == 1,
                "format 17 requires one head per plane, got nh=", sd.nh);
  }
  TORCH_CHECK(slots_per_object > 0, "slots per object must be positive, got ",
              slots_per_object);

  PreparedGroup group;
  group.layout.num_planes = sd.num_planes;
  // Per-segment budget: host-queried UB (query_ub_budget_bytes) or the
  // built-in default; the kernel receives the same ub_bytes.
  const int64_t ub_segment =
      ub_bytes > 0 ? ub_bytes / kvcache_ops::kBlockTransferQueueDepth
                   : kUbSegmentBytes;
  int64_t prefix = 0;
  for (int32_t p = 0; p < sd.num_planes; ++p) {
    const int64_t payload = sd.plane_slot_bytes[p];
    const int64_t block_stride = sd.plane_block_stride_bytes[p];
    TORCH_CHECK(payload > 0, "plane ", p, " payload must be positive, got ",
                payload);
    TORCH_CHECK(align_up32(payload) <= ub_segment, "plane ", p,
                " aligned row (", align_up32(payload),
                " bytes) exceeds the per-segment UB budget (", ub_segment, ")");
    const int64_t span = checked_mul(sd.bs, payload, "plane block span");
    TORCH_CHECK(block_stride >= span, "plane ", p, " block stride (",
                block_stride, ") is below the dense block span (", span,
                "); blocks would overlap");
    TORCH_CHECK(block_stride % kGmAlignBytes == 0, "plane ", p,
                " engine block stride (", block_stride,
                " bytes) must be a multiple of ", kGmAlignBytes,
                " for DataCopy alignment");
    // Address-range overflow probe: (nb - 1) * stride + span must stay
    // inside int64.
    checked_add(checked_mul(sd.nb - 1, block_stride, "plane address range"),
                span, "plane address range");
    if (!fmt17) {
      TORCH_CHECK(payload == scalar_row, "plane ", p, " payload (", payload,
                  ") must equal the full scalar row (", scalar_row,
                  ") for format ", static_cast<int>(engine_kv_format));
    }
    // LMC packed-row gap = bytes of the OTHER planes in the row; it feeds a
    // uint32 DataCopyPad field (engine-side gap is always 0: dense rows).
    TORCH_CHECK(scalar_row >= payload && scalar_row - payload <= kCopyGmGapMax,
                "plane ", p, " LMC row gap (", scalar_row - payload,
                ") exceeds the DataCopyPad GM gap range");
    const int64_t base_offset = sd.plane_engine_base_offset_bytes[p];
    TORCH_CHECK(base_offset >= 0, "plane ", p,
                " engine base offset must be non-negative, got ", base_offset);
    if (fmt2) {
      const int64_t reach =
          checked_add(base_offset, span, "format 2 plane reach");
      TORCH_CHECK(reach <= block_stride, "format 2 plane ", p,
                  " offset + block span (", reach,
                  ") exceeds the block stride (", block_stride, ")");
      // The probe above starts at offset 0. Format 2 also reaches base_offset.
      checked_add(checked_mul(sd.nb - 1, block_stride, "plane address range"),
                  reach, "plane address range");
    } else {
      TORCH_CHECK(base_offset == 0, "plane ", p,
                  " engine base offset is only valid for format 2, got ",
                  base_offset);
    }
    group.layout.planes[p] =
        kvcache_ops::PlaneLayout{payload, block_stride, fmt17 ? prefix : 0};
    group.layout.planes[p].engine_base_offset_bytes = base_offset;
    prefix = checked_add(prefix, payload, "LMC row prefix sum");
  }
  if (fmt2) {
    TORCH_CHECK(
        sd.plane_block_stride_bytes[0] == sd.plane_block_stride_bytes[1],
        "format 2 planes must share one block stride, got ",
        sd.plane_block_stride_bytes[0], " and ",
        sd.plane_block_stride_bytes[1]);
    const int64_t o0 = sd.plane_engine_base_offset_bytes[0];
    const int64_t o1 = sd.plane_engine_base_offset_bytes[1];
    const int64_t span0 =
        checked_mul(sd.bs, sd.plane_slot_bytes[0], "format 2 plane 0 span");
    const int64_t span1 =
        checked_mul(sd.bs, sd.plane_slot_bytes[1], "format 2 plane 1 span");
    const int64_t first = o0 <= o1 ? o0 : o1;
    const int64_t first_span = o0 <= o1 ? span0 : span1;
    const int64_t second = o0 <= o1 ? o1 : o0;
    TORCH_CHECK(second >= checked_add(first, first_span, "format 2 plane gap"),
                "format 2 K/V planes overlap: offsets ", o0, " and ", o1);
  }
  if (fmt17) {
    TORCH_CHECK(prefix == scalar_row, "sum of plane payloads (", prefix,
                ") must equal nh * hs * element_size (", scalar_row,
                ") for format 17");
  }

  group.layout.lmc_token_stride_bytes = scalar_row;
  group.layout.lmc_layer_stride_bytes =
      checked_mul(slots_per_object, scalar_row, "LMC layer stride bytes");
  const int64_t slab =
      checked_mul(sd.nl, group.layout.lmc_layer_stride_bytes, "LMC slab bytes");
  group.layout.lmc_object_bytes =
      checked_mul(separate ? 2 : 1, slab, "LMC object bytes");
  if (separate) {
    // 2LTD: K slab first, V slab after (planes[1] base = one full slab).
    group.layout.planes[1].lmc_base_offset_bytes = slab;
  }
  // LMC pages must not share 32B lines.
  TORCH_CHECK(
      checked_mul(sd.bs, scalar_row, "LMC page bytes") % kGmAlignBytes == 0,
      "BS * LMC row bytes (", sd.bs * scalar_row, ") must be a multiple of ",
      kGmAlignBytes, "; otherwise adjacent LMC pages share a 32B line");

  group.separate_plane = separate;
  group.one_pointer_per_layer = fmt2;
  group.nl = sd.nl;
  group.nb = sd.nb;
  group.bs = sd.bs;
  group.slots_per_object = slots_per_object;
  group.ub_bytes = ub_bytes;
  return group;
}

// ---------------------------------------------------------------------------
// validate_launch: per-transfer dynamic variables.
// ---------------------------------------------------------------------------

struct CheckedLaunch {
  int32_t total_blocks = 0;
  int32_t num_objects = 0;
  int32_t blocks_per_object = 0;
  int32_t skip_prefix_n_blocks = 0;
};

CheckedLaunch validate_launch(const PreparedGroup& group, int64_t total_blocks,
                              int num_objects, int64_t block_ids_offset,
                              int64_t block_ids_capacity,
                              int skip_prefix_n_blocks) {
  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);
  TORCH_CHECK(total_blocks >= 0, "total_blocks must be non-negative, got ",
              total_blocks);
  TORCH_CHECK(total_blocks % num_objects == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_objects (", num_objects,
              ")");
  const int64_t blocks = total_blocks / num_objects;
  TORCH_CHECK(
      checked_mul(blocks, group.bs, "blocks * bs") == group.slots_per_object,
      "blocks_per_object * block_size (", blocks * group.bs,
      ") must equal slots per object (", group.slots_per_object, ")");
  TORCH_CHECK(skip_prefix_n_blocks >= 0 && skip_prefix_n_blocks <= blocks,
              "skip_prefix_n_blocks (", skip_prefix_n_blocks,
              ") must be within [0, ", blocks, "]");
  // Subtraction form: offset + length itself may not overflow.
  TORCH_CHECK(block_ids_offset >= 0 && block_ids_offset <= block_ids_capacity,
              "block_ids_offset (", block_ids_offset, ") must be within [0, ",
              block_ids_capacity, "]");
  TORCH_CHECK(total_blocks <= block_ids_capacity - block_ids_offset,
              "block_ids slice [", block_ids_offset, ", ",
              block_ids_offset + total_blocks, ") exceeds block_ids capacity ",
              block_ids_capacity);
  TORCH_CHECK(total_blocks <= std::numeric_limits<int32_t>::max() &&
                  blocks <= std::numeric_limits<int32_t>::max(),
              "block counts exceed the kernel's int32 launch arguments");

  CheckedLaunch out;
  out.total_blocks = static_cast<int32_t>(total_blocks);
  out.num_objects = num_objects;
  out.blocks_per_object = static_cast<int32_t>(blocks);
  out.skip_prefix_n_blocks = skip_prefix_n_blocks;
  return out;
}

// ---------------------------------------------------------------------------
// Launch: one kernel launch per object.
// ---------------------------------------------------------------------------

void launch_prepared_objects(uint32_t aiv_num, void* stream,
                             const PreparedGroup& group,
                             uint8_t* paged_buffer_ptrs,
                             const std::vector<int64_t>& obj_device_ptrs,
                             int64_t* block_ids_base,
                             const CheckedLaunch& launch, bool to_engine) {
  // blockDim matches the ACTIVE work-item count (the kernel folds skip into
  // its index map). skip == blocks -> nothing to transfer; return early
  // because <<<0>>> would be an invalid launch.
  const int32_t plane_slots =
      group.separate_plane ? group.layout.num_planes : 1;
  const int64_t active_blocks =
      launch.blocks_per_object - launch.skip_prefix_n_blocks;
  const int64_t work =
      static_cast<int64_t>(group.nl) * plane_slots * active_blocks;
  if (work == 0) {
    return;
  }
  const uint32_t blockDim =
      static_cast<uint32_t>(std::min<int64_t>(aiv_num, work));
  for (int32_t i = 0; i < launch.num_objects; ++i) {
    uint8_t* engine_block_ids = reinterpret_cast<uint8_t*>(
        block_ids_base + static_cast<int64_t>(i) * launch.blocks_per_object);
    kvcache_ops::multi_layer_block_transfer_kernel(
        blockDim, stream, paged_buffer_ptrs,
        reinterpret_cast<uint8_t*>(obj_device_ptrs[i]), engine_block_ids,
        launch.blocks_per_object, launch.skip_prefix_n_blocks, group.nl,
        group.nb, group.bs, group.separate_plane, group.layout, to_engine,
        group.ub_bytes, group.one_pointer_per_layer);
  }
}

// ---------------------------------------------------------------------------
// LMC object device-VA resolution and staging (direct entry).
// ---------------------------------------------------------------------------

// CUDA UVA lets the kernel consume host data_ptr()s. Ascend DataCopy needs a
// device VA: registered aclrtMallocHost (get_device_ptr) or a true NPU
// allocation. MixedMemoryAllocator SHM L1 is neither — stage on-device and
// aclrtMemcpy.
bool is_npu_memory_ptr(void* ptr) {
  aclrtPtrAttributes attributes{};
  const aclError ret = aclrtPointerGetAttributes(ptr, &attributes);
  if (ret != ACL_ERROR_NONE) {
    return false;
  }
  return attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE;
}

std::vector<int64_t> device_lmc_ptrs(const std::vector<int64_t>& ptrs) {
  std::vector<int64_t> out;
  out.reserve(ptrs.size());
  for (int64_t p : ptrs) {
    void* mapped =
        get_device_ptr(reinterpret_cast<void*>(static_cast<uintptr_t>(p)));
    out.push_back(mapped != nullptr ? reinterpret_cast<int64_t>(mapped) : p);
  }
  return out;
}

struct HostStage {
  torch::Tensor buf;
  int64_t host_ptr = 0;
  size_t nbytes = 0;
};

// Resolve LMCache object pointers to device to device virtual addresses.
// SHM/host objects that are neither registered aclrtMallocHost nor NPU memory
// get an on-device staging tensor (HostStage); the caller memcpy-asyncs around
// the kernel. Allocations happen BEFORE any enqueue; the caller's cache context
// synchronizes the stream before releasing them, so the staging tensors stay
// valid for the async copies.
struct PreparedLmcPtrs {
  std::vector<int64_t> kernel_obj_ptrs;
  std::vector<HostStage> host_stages;
};

PreparedLmcPtrs prepare_lmc_ptrs(const std::vector<int64_t>& lmc_obj_ptrs,
                                 const torch::Device& device,
                                 const PreparedGroup& group) {
  PreparedLmcPtrs prepared;
  prepared.kernel_obj_ptrs.reserve(lmc_obj_ptrs.size());
  const auto staging_opts =
      torch::TensorOptions().dtype(torch::kUInt8).device(device);
  // One capacity formula for every layout.
  const int64_t object_bytes = group.layout.lmc_object_bytes;
  TORCH_CHECK(object_bytes > 0, "LMCache object byte size must be positive");
  for (int64_t p : lmc_obj_ptrs) {
    void* raw = reinterpret_cast<void*>(static_cast<uintptr_t>(p));
    void* mapped = get_device_ptr(raw);
    if (mapped != nullptr) {
      prepared.kernel_obj_ptrs.push_back(reinterpret_cast<int64_t>(mapped));
      continue;
    }
    if (is_npu_memory_ptr(raw)) {
      prepared.kernel_obj_ptrs.push_back(p);
      continue;
    }
    // A torch-pinned host pointer may not have a mapping in our registry.
    HostStage stage;
    stage.buf = torch::empty({object_bytes}, staging_opts);
    stage.host_ptr = p;
    stage.nbytes = static_cast<size_t>(object_bytes);
    prepared.kernel_obj_ptrs.push_back(
        reinterpret_cast<int64_t>(stage.buf.data_ptr()));
    prepared.host_stages.push_back(std::move(stage));
  }
  return prepared;
}

// Enqueue HostStage memcpys + the block kernel(s) on ``stream``: H2D stages
// before the kernel, D2H stages after, and a partial D2H store first reads
// the host object into staging so the untouched prefix is preserved when
// the whole object is written back.
int enqueue_block_transfer(void* stream, uint32_t aiv_num,
                           uint8_t* paged_buffer_ptrs,
                           const PreparedGroup& group,
                           const PreparedLmcPtrs& prepared,
                           int64_t* block_ids_base, const CheckedLaunch& launch,
                           bool to_engine) {
  if (to_engine || launch.skip_prefix_n_blocks > 0) {
    for (const auto& stage : prepared.host_stages) {
      const aclError ret = aclrtMemcpyAsync(
          stage.buf.data_ptr(), stage.nbytes,
          reinterpret_cast<const void*>(static_cast<uintptr_t>(stage.host_ptr)),
          stage.nbytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
      if (ret != ACL_ERROR_NONE) {
        return static_cast<int>(ret);
      }
    }
  }
  launch_prepared_objects(aiv_num, stream, group, paged_buffer_ptrs,
                          prepared.kernel_obj_ptrs, block_ids_base, launch,
                          to_engine);
  if (!to_engine) {
    for (const auto& stage : prepared.host_stages) {
      const aclError ret = aclrtMemcpyAsync(
          reinterpret_cast<void*>(static_cast<uintptr_t>(stage.host_ptr)),
          stage.nbytes, stage.buf.data_ptr(), stage.nbytes,
          ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        return static_cast<int>(ret);
      }
    }
  }
  return 0;
}

}  // namespace

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  // Device guard first: the UB query below needs the current device's SoC.
  const c10::OptionalDeviceGuard device_guard(device);
  // --- Static geometry (prepare_group) + dynamic variables (validate_launch)
  const PreparedGroup group =
      prepare_group(shape_desc, engine_kv_format, lmcache_chunk_size,
                    query_ub_budget_bytes());
  const int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  const int64_t total_blocks = block_ids.size(0);
  const CheckedLaunch launch = validate_launch(
      group, total_blocks, num_objects,
      /*block_ids_offset=*/0,
      /*block_ids_capacity=*/total_blocks, skip_prefix_n_blocks);

  TORCH_CHECK(paged_buffer_ptrs_tensor.scalar_type() == at::kLong,
              "paged_buffer_ptrs_tensor must be int64");
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_privateuseone(),
              "paged_buffer_ptrs_tensor must live on the NPU");
  TORCH_CHECK(paged_buffer_ptrs_tensor.dim() == 1,
              "paged_buffer_ptrs_tensor must be one-dimensional");
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_contiguous(),
              "paged_buffer_ptrs_tensor must be contiguous");
  if (group.one_pointer_per_layer) {
    TORCH_CHECK(paged_buffer_ptrs_tensor.numel() == group.nl,
                "paged_buffer_ptrs_tensor must contain nl pointers: expected ",
                group.nl, ", got ", paged_buffer_ptrs_tensor.numel());
  } else {
    // Pointer table: one device pointer per (layer, physical plane).
    const int64_t expected_ptrs =
        static_cast<int64_t>(group.layout.num_planes) * group.nl;
    TORCH_CHECK(paged_buffer_ptrs_tensor.numel() == expected_ptrs,
                "paged_buffer_ptrs_tensor must contain num_planes * nl "
                "pointers: expected ",
                expected_ptrs, ", got ", paged_buffer_ptrs_tensor.numel());
  }
  TORCH_CHECK(block_ids.is_privateuseone(), "block_ids must live on the NPU");
  TORCH_CHECK(block_ids.scalar_type() == at::kLong,
              "block_ids must have dtype int64");
  TORCH_CHECK(block_ids.dim() == 1, "block_ids must be one-dimensional");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids must be contiguous");

  const bool to_engine = (direction == TransferDirection::H2D);

  PreparedLmcPtrs prepared =
      prepare_lmc_ptrs(lmcache_objects_ptrs, device, group);

  uint8_t* paged_buffer_ptrs =
      static_cast<uint8_t*>(paged_buffer_ptrs_tensor.data_ptr());
  int64_t* block_ids_base = block_ids.data_ptr<int64_t>();

  // .stream() drains torch_npu's ACL task queue before returning the raw
  // stream. The drain is per-call by contract: each call must keep its direct
  // enqueues ordered after torch ops issued since the last drain. This op can
  // be invoked in a loop (per-block fallback paths in callers); the first
  // call flushes the queue and the following ones find it empty, so the
  // repeated drain is a cheap empty-queue check.
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  // Direct launch (CUDA cudaLaunchKernel style): the kernel + staging memcpys
  // below are enqueued on the calling thread, no OpCommand / queue-thread hop.
  const char* socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  const uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
  const int ret =
      enqueue_block_transfer(stream, aiv_num, paged_buffer_ptrs, group,
                             prepared, block_ids_base, launch, to_engine);
  TORCH_CHECK(ret == ACL_ERROR_NONE,
              "multi_layer_block_transfer staging failed, aclrtMemcpyAsync "
              "ret=",
              ret);
}

void lmcache_memcpy_async_on_stream(uintptr_t dest, uintptr_t src,
                                    size_t nbytes, TransferDirection direction,
                                    size_t host_buffer_offset,
                                    size_t host_buffer_alignments,
                                    aclrtStream stream) {
  TORCH_CHECK(host_buffer_alignments > 0 &&
                  (host_buffer_alignments & (host_buffer_alignments - 1)) == 0,
              "host_buffer_alignments must be a non-zero power of two");

  size_t offset = 0;
  const size_t mask = host_buffer_alignments - 1;
  const aclrtMemcpyKind kind = (direction == TransferDirection::H2D)
                                   ? ACL_MEMCPY_HOST_TO_DEVICE
                                   : ACL_MEMCPY_DEVICE_TO_HOST;

  while (offset < nbytes) {
    const size_t aligned_area_end =
        ((offset + host_buffer_offset) & ~mask) + host_buffer_alignments;
    const size_t real_end =
        std::min<size_t>(host_buffer_offset + nbytes, aligned_area_end);
    const size_t max_nbytes = real_end - offset - host_buffer_offset;

    const aclError ret = aclrtMemcpyAsync(
        reinterpret_cast<void*>(dest + offset), max_nbytes,
        reinterpret_cast<const void*>(src + offset), max_nbytes, kind, stream);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMemcpyAsync failed: ret=", ret);

    offset += max_nbytes;
  }
}

void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments) {
  lmcache_memcpy_async_on_stream(dest, src, nbytes, direction,
                                 host_buffer_offset, host_buffer_alignments,
                                 c10_npu::getCurrentNPUStream().stream());
}

void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps) {
  // Set the device guard once for the whole plan so every staging copy and
  // kernel launch below is enqueued on this device's current stream, in
  // order (mirrors upstream execute_object_group_transfer).
  const c10::OptionalDeviceGuard device_guard(device);
  const bool is_h2d = (direction == TransferDirection::H2D);
  TORCH_CHECK(device.is_privateuseone(), "device must be an NPU device");
  TORCH_CHECK(host_buffer_alignment > 0 &&
                  (host_buffer_alignment & (host_buffer_alignment - 1)) == 0,
              "host_buffer_alignment must be a non-zero power of two, got ",
              host_buffer_alignment);

  // --- Whole-plan pre-validation: every group is prepared and
  // every launch is validated before ANY staging copy or kernel launch is
  // enqueued. A failure here means "nothing started".
  const int64_t ub_bytes = query_ub_budget_bytes();
  std::vector<PreparedGroup> groups;
  groups.reserve(kernel_group_specs.size());
  for (const auto& spec : kernel_group_specs) {
    groups.push_back(prepare_group(spec.shape_desc, spec.engine_kv_format,
                                   spec.lmcache_chunk_size, ub_bytes));
  }

  struct PreparedLaunch {
    int32_t group_idx;
    CheckedLaunch checked;
    int64_t block_ids_offset;
  };
  std::vector<std::vector<PreparedLaunch>> prepared_steps;
  prepared_steps.reserve(batch_steps.size());
  for (const auto& step : batch_steps) {
    for (const auto& copy : step.staging) {
      // Raw external pointers carry no derivable allocation capacity; the
      // plan builder owns pointer/size validity. What can be
      // checked cheaply here is checked.
      TORCH_CHECK(copy.nbytes > 0, "StagingCopy nbytes must be positive");
      TORCH_CHECK(copy.dest != 0 && copy.src != 0,
                  "StagingCopy pointers must be non-null");
    }
    std::vector<PreparedLaunch> launches;
    launches.reserve(step.launches.size());
    for (const auto& launch : step.launches) {
      TORCH_CHECK(
          launch.group_idx >= 0 &&
              launch.group_idx < static_cast<int>(kernel_group_specs.size()),
          "LaunchVar.group_idx out of range: ", launch.group_idx);
      const KernelGroupSpec& spec = kernel_group_specs[launch.group_idx];
      TORCH_CHECK(launch.num_objects <=
                      static_cast<int>(spec.lmcache_objects_ptrs.size()),
                  "LaunchVar.num_objects (", launch.num_objects,
                  ") exceeds available temp buffers (",
                  spec.lmcache_objects_ptrs.size(), ")");
      PreparedLaunch prepared;
      prepared.group_idx = launch.group_idx;
      prepared.checked =
          validate_launch(groups[launch.group_idx], launch.total_blocks,
                          launch.num_objects, launch.block_ids_offset,
                          spec.block_ids_capacity, launch.skip_prefix_n_blocks);
      prepared.block_ids_offset = launch.block_ids_offset;
      launches.push_back(prepared);
    }
    prepared_steps.push_back(std::move(launches));
  }

  const char* socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  const uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  // Direct launch (CUDA cudaLaunchKernel style): .stream() above drains
  // torch_npu's ACL task queue first, so every staging copy and kernel launch
  // below is enqueued in Python-call order without a queue-thread hop. The
  // pointer table, block IDs, staging buffers and engine plane views are
  // owned by the caller's cache context, which synchronizes the stream before
  // releasing them, so the raw pointers below stay valid.
  const auto do_staging = [&](const std::vector<StagingCopy>& staging) {
    for (const auto& copy : staging) {
      lmcache_memcpy_async_on_stream(copy.dest, copy.src, copy.nbytes,
                                     direction, copy.host_offset,
                                     host_buffer_alignment, stream);
    }
  };

  for (size_t step_idx = 0; step_idx < batch_steps.size(); ++step_idx) {
    // H2D stages CPU->NPU temp buffers before the kernel reads them; D2H
    // stages NPU->CPU after the kernel writes them. The per-step ordering
    // must be preserved because temp buffers are reused across steps.
    if (is_h2d) {
      do_staging(batch_steps[step_idx].staging);
    }
    for (const auto& launch : prepared_steps[step_idx]) {
      const KernelGroupSpec& spec = kernel_group_specs[launch.group_idx];
      const PreparedGroup& group = groups[launch.group_idx];
      std::vector<int64_t> obj_device_ptrs =
          device_lmc_ptrs(std::vector<int64_t>(
              spec.lmcache_objects_ptrs.begin(),
              spec.lmcache_objects_ptrs.begin() + launch.checked.num_objects));
      int64_t* block_ids_base = reinterpret_cast<int64_t*>(
          spec.block_ids_base +
          static_cast<uintptr_t>(launch.block_ids_offset) * sizeof(int64_t));
      launch_prepared_objects(
          aiv_num, stream, group,
          reinterpret_cast<uint8_t*>(spec.paged_buffer_ptrs), obj_device_ptrs,
          block_ids_base, launch.checked, is_h2d);
    }
    if (!is_h2d) {
      do_staging(batch_steps[step_idx].staging);
    }
  }
}
