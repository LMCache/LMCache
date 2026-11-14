// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Host/device shared POD for the MP block-level KV transfer.
//
// The host parser (csrc/mp_mem_kernels.cpp, prepare_group) resolves a
// PageBufferShapeDesc plus EngineKVFormat into this flat byte layout; the
// device kernel consumes it verbatim and never re-interprets the format.
// All sizes/offsets are in bytes (int64): planes may mix dtypes, and
// engine pools may exceed 32-bit element counts.
//
// Layout contracts:
//   - Engine plane p: [NB, BS, NH, HS] with dense token rows, so the
//     engine-side token step equals payload_bytes. Only the block axis may
//     carry padding, expressed by engine_block_stride_bytes.
//   - LMC side (formats 13/17): one packed row per token. Format 13 has one
//     fused plane; format 17 has lmc_token_stride_bytes = sum(payload_bytes)
//     and plane p sits at its prefix-sum column offset inside the row.
//   - Format 16: two independent slabs of nl * lmc_layer_stride_bytes;
//     planes[1].lmc_base_offset_bytes == nl * lmc_layer_stride_bytes.

namespace kvcache_ops {

constexpr int kMaxPlanes = 4;

// Per-AIV UB staging budget, a conservative cross-SoC floor (950 has 248KB
// physical UB per vector core). Bounds one token row, not a whole block;
// full budget contract in block_transfer_contract.md section 3.
constexpr int64_t kBlockTransferUbBytes = 128 * 1024;

constexpr int32_t kBlockTransferQueueDepth = 1;

// DataCopyExtParams::blockCount hardware upper bound for the DataCopyPad row
// path ([1, 4095] per the AscendC API reference). The kernel clamps
// rows-per-segment to this value: a thin plane (payload <= 32B) gives
// fit = kBlockTransferUbBytes / 32B = 4096 rows, one past the limit.
constexpr int32_t kMaxRowsPerDataCopyPad = 4095;

struct PlaneLayout {
  int64_t payload_bytes;             // valid bytes per token slot
  int64_t engine_block_stride_bytes; // dim-0 block step on the engine side
  int64_t lmc_base_offset_bytes;     // plane start inside the LMC object
  // Format 2 only: bytes from the layer base pointer to this plane.
  // Formats 16/17 pass one pointer per plane, so this stays 0.
  int64_t engine_base_offset_bytes = 0;
};

struct BlockTransferLayout {
  int32_t num_planes;             // 1..kMaxPlanes (physical planes/layer)
  int64_t lmc_token_stride_bytes; // LMC packed row width (all planes)
  int64_t lmc_layer_stride_bytes; // bytes per layer = slots_per_object * row
  int64_t lmc_object_bytes;       // total object size
  PlaneLayout planes[kMaxPlanes];
};

} // namespace kvcache_ops
