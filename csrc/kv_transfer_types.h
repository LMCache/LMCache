// SPDX-License-Identifier: Apache-2.0

#pragma once

// Backend-agnostic transfer descriptors, free of any vendor runtime headers so
// every backend (CUDA, ROCm, MUSA, SYCL/XPU, ...) shares one definition.

#include "engine_kv_format.h"  // EngineKVFormat + its classification predicates

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

// Declared layout of the LMCache-side buffer passed to the in-process
// multi-layer transfer entry points. Fused-packed engine formats admit two
// LMCache-side layouts that cannot be told apart reliably from the buffer
// shape, so callers must declare which one they pass; the axis does not
// exist for any other format (those must pass UNSPECIFIED).
enum class MemObjKVLayout : int {
  UNSPECIFIED = 0,
  // True split-KV object: [2, num_layers, num_tokens, num_heads * head_size].
  SPLIT_KV_2LTD = 1,
  // Packed single plane mirroring the engine's fused rows:
  // [1, num_layers, num_tokens, num_heads * 2 * head_size].
  FUSED_PACKED = 2,
};
