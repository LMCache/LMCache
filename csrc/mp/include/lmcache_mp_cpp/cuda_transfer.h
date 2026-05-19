// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "lmcache_mp_cpp/tiered_cache.h"

#include <cstdint>
#include <string>
#include <vector>

namespace lmcache::mp {

struct KvTensorMetadata {
  std::string kind;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint64_t> stride;
  std::uint64_t storage_offset = 0;
  std::string device_uuid;
  std::uint64_t storage_bytes = 0;
  std::uint64_t storage_offset_bytes = 0;
  std::vector<std::uint8_t> ipc_handle;
  std::vector<std::uint8_t> event_handle;
  bool event_sync_required = false;
};

struct KvTransferRequest {
  std::vector<KvTensorMetadata> tensors;
  std::vector<std::string> object_keys;
  std::vector<std::uint64_t> gpu_block_ids;
  std::vector<std::uint8_t> input_event_handle;
  std::uint32_t chunk_size = 0;
  std::uint32_t logical_block_size = 0;
  std::uint64_t skip_first_n_tokens = 0;
  std::string kv_layout;
  bool trt_llm_layout_hints = false;
  std::uint64_t trt_llm_num_kv_heads = 0;
  std::uint64_t trt_llm_tokens_per_block = 0;
  std::uint64_t trt_llm_head_dim = 0;
  bool enable_gpu_hot_cache = false;
};

struct KvTransferStats {
  std::uint64_t bytes = 0;
  std::uint64_t cuda_memcpy_calls = 0;
  std::uint64_t cuda_kernel_calls = 0;
  std::uint64_t wait_event_us = 0;
  std::uint64_t open_tensors_us = 0;
  std::uint64_t copy_us = 0;
  std::uint64_t cache_us = 0;
  std::uint64_t completion_event_us = 0;
};

struct KvTransferResult {
  bool success = false;
  std::vector<std::uint8_t> completion_event_handle;
  std::string error;
  KvTransferStats stats;
};

struct CudaTransferDeviceCacheStats {
  std::uint64_t entries = 0;
  std::uint64_t bytes = 0;
};

struct KvChecksumRequest {
  std::vector<KvTensorMetadata> tensors;
  std::vector<std::uint64_t> gpu_block_ids;
  std::uint32_t chunk_blocks = 0;
  std::string kv_layout;
  bool layerwise = false;
  bool trt_llm_layout_hints = false;
  std::uint64_t trt_llm_num_kv_heads = 0;
  std::uint64_t trt_llm_tokens_per_block = 0;
  std::uint64_t trt_llm_head_dim = 0;
};

struct KvChecksumResult {
  bool success = false;
  std::uint64_t num_chunks = 0;
  std::vector<std::string> chunk_checksums;
  std::vector<std::vector<std::string>> layerwise_chunk_checksums;
  std::string error;
};

bool NativeCudaTransferEnabled();

CudaTransferDeviceCacheStats GetCudaTransferDeviceCacheStats();

KvTransferResult StoreKvChunksFromCuda(const KvTransferRequest& request,
                                       LmcacheMpCppCache* cache);

KvTransferResult StoreKvChunksToCudaHotCacheAsync(
    const KvTransferRequest& request);

bool WarmCudaTransferTensorHandles(
    const std::vector<KvTensorMetadata>& tensors, std::string* error);

KvTransferResult RetrieveKvChunksToCuda(const KvTransferRequest& request,
                                        LmcacheMpCppCache* cache);

KvChecksumResult ChecksumKvCacheBlocksFromCuda(
    const KvChecksumRequest& request);

void ReleaseCudaTransferEvents();

bool CudaTransferDeviceChunkReady(const std::string& key);

bool ClearCudaTransferDeviceCache(std::string* error);

}  // namespace lmcache::mp
