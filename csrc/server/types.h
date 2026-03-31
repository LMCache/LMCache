// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — Shared type definitions
//
// All wire types (sent over ZMQ msgpack) and internal types shared between
// components live here.  Keeps torch/ATen out of the common include path;
// DType is our own lightweight enum converted to at::ScalarType only in
// tensor_bridge.h.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace lmcache {
namespace server {

// ============================================================================
// RequestType — exact values matching Python enum.auto() (1-based)
// ============================================================================

enum class RequestType : int {
  // Engine operations
  REGISTER_KV_CACHE = 1,
  UNREGISTER_KV_CACHE = 2,
  STORE = 3,
  RETRIEVE = 4,
  LOOKUP = 5,
  QUERY_PREFETCH_STATUS = 6,
  QUERY_PREFETCH_LOOKUP_HITS = 7,
  FREE_LOOKUP_LOCKS = 8,
  END_SESSION = 9,

  // Controller operations
  CLEAR = 10,
  GET_CHUNK_SIZE = 11,
  PING = 12,

  // Debug operations
  NOOP = 13,

  // Blend operations
  CB_REGISTER_KV_CACHE = 14,
  CB_UNREGISTER_KV_CACHE = 15,
  CB_STORE_PRE_COMPUTED = 16,
  CB_LOOKUP_PRE_COMPUTED = 17,
  CB_RETRIEVE_PRE_COMPUTED = 18,
  CB_STORE_FINAL = 19,

  // Blend V2 operations
  CB_LOOKUP_PRE_COMPUTED_V2 = 20,
  CB_RETRIEVE_PRE_COMPUTED_V2 = 21,
};

// ============================================================================
// HandlerType — how a request handler is executed
// ============================================================================

enum class HandlerType : int {
  SYNC = 1,      // Fast, runs in main loop
  BLOCKING = 2,  // May block, runs in thread pool
};

// ============================================================================
// RequestUID — opaque request identifier on the wire
// ============================================================================

using RequestUID = int64_t;

// ============================================================================
// DType — lightweight dtype enum (avoids #include <torch/torch.h> everywhere)
// ============================================================================

enum class DType : int {
  Float16 = 0,
  BFloat16 = 1,
  Float32 = 2,
  Float8E4M3FN = 3,
  Float8E5M2 = 4,
  Int8 = 5,
  Int32 = 6,
  Int64 = 7,
};

/// Size in bytes of one element of the given dtype.
inline size_t dtype_size(DType dt) {
  switch (dt) {
    case DType::Float16:
    case DType::BFloat16:
      return 2;
    case DType::Float32:
    case DType::Int32:
      return 4;
    case DType::Int64:
      return 8;
    case DType::Float8E4M3FN:
    case DType::Float8E5M2:
    case DType::Int8:
      return 1;
  }
  return 0;
}

// ============================================================================
// ObjectKey — unique identifier for a KV cache object in distributed storage
// ============================================================================

struct ObjectKey {
  std::vector<uint8_t>
      chunk_hash;  // Content hash bytes (typically 32 for blake3)
  std::string model_name;
  int32_t kv_rank;

  bool operator==(const ObjectKey& other) const {
    return chunk_hash == other.chunk_hash && model_name == other.model_name &&
           kv_rank == other.kv_rank;
  }

  /// Compute kv_rank from parallelism dimensions.
  /// Each number uses 8 bits, packed as:
  ///   (world_size << 24) | (global_rank << 16) |
  ///   (local_world_size << 8) | local_rank
  static inline int32_t compute_kv_rank(int world_size, int global_rank,
                                        int local_world_size, int local_rank) {
    return (world_size << 24) | (global_rank << 16) | (local_world_size << 8) |
           local_rank;
  }
};

struct ObjectKeyHash {
  size_t operator()(const ObjectKey& k) const {
    // FNV-1a over chunk_hash + model_name + kv_rank
    size_t h = 14695981039346656037ULL;
    for (uint8_t b : k.chunk_hash) {
      h ^= b;
      h *= 1099511628211ULL;
    }
    for (char c : k.model_name) {
      h ^= static_cast<uint8_t>(c);
      h *= 1099511628211ULL;
    }
    h ^= static_cast<size_t>(k.kv_rank);
    h *= 1099511628211ULL;
    return h;
  }
};

/// Convert an IPCCacheEngineKey + chunk_hashes to a list of ObjectKeys.
/// When worker_id < 0, expands each chunk hash to all workers in world_size.
std::vector<ObjectKey> ipc_key_to_object_keys(
    const std::string& model_name, int world_size,
    int worker_id,  // -1 means expand to all workers
    const std::vector<std::vector<uint8_t>>& chunk_hashes);

// ============================================================================
// IPCCacheEngineKey — token-based cache key sent over ZMQ
// ============================================================================

struct IPCCacheEngineKey {
  std::string model_name;
  int32_t world_size;
  int32_t worker_id;  // -1 means None (lookup mode)

  std::vector<int32_t> token_ids;
  int32_t start;
  int32_t end;

  // Session tracking — not part of cache identity
  std::string request_id;
};

// ============================================================================
// MemoryLayoutDesc — shape + dtype descriptors for memory objects
// ============================================================================

struct ShapeDesc {
  std::vector<int64_t> dims;
};

struct MemoryLayoutDesc {
  std::vector<ShapeDesc> shapes;
  std::vector<DType> dtypes;
};

// ============================================================================
// CudaIpcTensorDesc — structured replacement for Python CudaIPCWrapper
//
// Binary layout for zero-copy decode.  Transition: the C++ decoder handles
// both the legacy pickle Ext(1) format and the new structured format.
// ============================================================================

struct CudaIpcTensorDesc {
  // The full torch serialized IPC handle blob from _share_cuda_()[1]
  // (typically 66 bytes in PyTorch 2.10+/CUDA 12.x, 64 in older)
  // Passed to c10::cuda::CUDACachingAllocator::getIpcDevPtr()
  std::vector<uint8_t> ipc_handle_blob;

  // Legacy 64-byte raw cudaIpcMemHandle_t (for fallback)
  uint8_t ipc_handle[64];

  // Storage size in bytes (handle[2] from _share_cuda_)
  int64_t storage_size_bytes;

  DType dtype;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  int64_t storage_offset;

  // GPU UUID string (e.g. "GPU-xxxxxxxx-...")
  std::string device_uuid;
};

// ============================================================================
// L1Error — result codes for L1 slab storage operations
// ============================================================================

enum class L1Error : int {
  SUCCESS = 0,
  KEY_NOT_EXIST = 1,
  KEY_NOT_READABLE = 2,
  KEY_NOT_WRITABLE = 3,
  KEY_IN_WRONG_STATE = 4,
  KEY_IS_LOCKED = 5,
  OUT_OF_MEMORY = 6,
};

// ============================================================================
// MemorySlabRef — zero-copy reference into L1 slab
// ============================================================================

struct MemorySlabRef {
  void* data;         // Pointer into the mmap'd slab
  size_t size_bytes;  // Size of this object
  MemoryLayoutDesc layout;
};

// ============================================================================
// PrefetchHandle — tracks a prefetch request from submission to completion
// ============================================================================

struct PrefetchHandle {
  int64_t prefetch_request_id;  // Opaque L2 tracking ID (-1 if no L2 request)
  std::string external_request_id;  // Caller's request ID for tracing
  int64_t l1_prefix_hit_count;      // Leading keys already in L1 at submit time
  int64_t total_requested_keys;     // Total number of keys originally requested
  double submit_time;               // Monotonic timestamp (seconds)
};

// ============================================================================
// CBMatchResult — sub-sequence match from BlendTokenRangeMatcher
// ============================================================================

struct CBMatchResult {
  int32_t old_st;
  int32_t old_ed;
  int32_t cur_st;
  int32_t cur_ed;
  std::vector<uint8_t> hash;  // Token hash bytes from registration
};

// ============================================================================
// compute_extra_count — MLA multi-reader locking logic
//
// Non-MLA: each TP worker owns a distinct KV shard, extra_count = 0.
// MLA: all TP workers share the same object, extra_count = tp_size - 1.
// Detection: tp > world_size means MLA (world_size was divided by tp).
// ============================================================================

inline int compute_extra_count(int tp_size, int world_size) {
  int tp = (tp_size > 1) ? tp_size : world_size;
  return (tp > world_size) ? (tp - 1) : 0;
}

}  // namespace server
}  // namespace lmcache
