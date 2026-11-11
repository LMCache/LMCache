// SPDX-License-Identifier: Apache-2.0

#ifndef MULTI_LAYER_BLOCK_MEM_KERNELS_H
#define MULTI_LAYER_BLOCK_MEM_KERNELS_H

#include "block_transfer_layout.h"

namespace kvcache_ops {

// Host-side launcher for the block-level multi-layer KV transfer kernel.
// One launch moves one LMCache memory object
//
// Layout semantics per engine KV format (see block_transfer_contract.md):
//   2  (NL_X_NB_TWO_BS_NH_HS): one [NB, 2, BS, NH, HS] tensor per layer.
//       The pointer table is one base data_ptr() per layer. Plane p is
//       base + plane.engine_base_offset_bytes. The planes are axis-1 slabs
//       (semantic K/V, or the two halves of an opaque sub-paged page).
//       LMC layout matches format 16.
//   16 (NL_X_TWO_X_NB_BS_NH_HS): per-layer (K, V) equal-width NHD tensors
//       -> LMC [2, L, T, NH*HS] (2LTD: K slab first, V slab after); work
//       items are (layer, plane, block) since K/V live in disjoint slabs.
//   17 (NL_X_NP_X_NB_BS_ONE_HS): per-layer tuple of 1..4 single-head planes
//       (MLA/DSA/DSv4) -> LMC [L, T, sum(payload)] packed byte rows; work
//       items are (layer, block) and one AIV core moves every plane of the
//       page, so no two cores ever share a 32B LMC line.
//
// Pointer table layout:
//   format 2: ptrs[layer], plus PlaneLayout::engine_base_offset_bytes;
//   formats 16/17: ptrs[layer * num_planes + plane], each entry the plane
//   tensor view's data_ptr().

void multi_layer_block_transfer_kernel(
    uint32_t block_dim,            // min(AIV cores, ACTIVE work items =
                                   //   nl * plane_slots * (blocks - skip))
    void* stream,                  // aclrtStream (current NPU stream)
    uint8_t* paged_buffer_ptrs,    // interleaved pointer table (see above)
    uint8_t* lmcache_obj,          // LMC object device VA (maybe staged)
    uint8_t* engine_block_ids,     // int64 array: logical -> engine block
    int32_t num_blocks_per_object, // == lmcache_chunk_size / bs (host-checked)
    int32_t skip_prefix_n_blocks,  // leading blocks to skip (prefix dedup);
                                   //   == blocks -> no launch at all
    int32_t nl,                    // layers in this group
    int32_t nb,                    // engine pool block count (ID upper bound)
    int32_t bs,                    // tokens/slots per block
    bool separate_plane,           // true: (layer, plane, block) work items
    BlockTransferLayout layout,    // full byte layout (block_transfer_layout.h)
    bool to_engine,                // H2D = LMC->engine (retrieve), D2H =
                                   // engine->LMC (store); device-to-device
    int64_t ub_bytes = 0,          // usable UB staging budget, host-queried
                                   //   via GetCoreMemSize(CoreMemType::UB);
                                   //   0 = built-in default
    bool one_pointer_per_layer = false); // format 2: one base pointer per layer
} // namespace kvcache_ops

#endif // MULTI_LAYER_BLOCK_MEM_KERNELS_H
