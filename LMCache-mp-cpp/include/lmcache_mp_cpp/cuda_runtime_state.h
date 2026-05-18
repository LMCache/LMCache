// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "lmcache_mp_cpp/cuda_transfer.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#if LMCACHE_ENABLE_CUDA
  #include <cuda_runtime_api.h>
#endif

namespace lmcache::mp {

// CUDA IPC handles, interprocess events, and device selection are process-global
// runtime state. Keep their ownership out of the copy-planning code so transfer
// logic can stay focused on byte movement.
std::mutex& CudaTransferMutex();

#if LMCACHE_ENABLE_CUDA
std::string CudaErrorString(cudaError_t err, const std::string& op);
bool CheckCuda(cudaError_t err, const std::string& op, std::string* error);

bool WaitForCudaInputEvent(const std::vector<std::uint8_t>& event_handle,
                           std::string* error);
bool WaitForCudaInputEventOnStream(
    const std::vector<std::uint8_t>& event_handle, void* cuda_stream,
    std::string* error);
bool OpenCudaTensorMemory(const KvTensorMetadata& metadata, void** ptr,
                          std::string* error);
void BestEffortSetCudaDeviceForUuid(const std::string& device_uuid);
void* CudaTransferStream(std::string* error);

bool EnsureCudaDeviceChunk(const std::string& key, std::uint64_t size,
                           void** ptr, std::string* error);
bool FindCudaDeviceChunk(const std::string& key, std::uint64_t size,
                         void** ptr);
void EraseCudaDeviceChunks(const std::vector<std::string>& keys);
bool MarkCudaDeviceChunksReadyOnStream(const std::vector<std::string>& keys,
                                       void* cuda_stream, std::string* error);
KvTransferResult MakeCudaCompletionEvent();
KvTransferResult MakeCudaCompletionEventOnStream(void* cuda_stream);
#endif

bool CudaDeviceChunkReady(const std::string& key);
void ClearCudaDeviceChunks();
CudaTransferDeviceCacheStats GetCudaRuntimeDeviceCacheStats();
void ReleaseCudaRuntimeState();

}  // namespace lmcache::mp
