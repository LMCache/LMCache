// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/cuda_runtime_state.h"

#include "lmcache_mp_cpp/cuda_metadata.h"

#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#if LMCACHE_ENABLE_CUDA
  #include <cstdio>
#endif

namespace lmcache::mp {
namespace {

#if LMCACHE_ENABLE_CUDA

struct DeviceChunk {
  void* ptr = nullptr;
  std::uint64_t size = 0;
  bool ready = false;
  cudaEvent_t ready_event = nullptr;
};

std::mutex g_events_mu;
std::mutex g_ipc_handles_mu;
std::mutex g_device_uuid_mu;
std::mutex g_device_chunks_mu;
std::mutex g_transfer_stream_mu;
std::vector<cudaEvent_t> g_completion_events;
std::unordered_map<std::string, void*> g_ipc_handles;
std::unordered_map<std::string, int> g_device_uuid_to_index;
std::unordered_map<std::string, DeviceChunk> g_device_chunks;
std::string g_current_device_uuid;
cudaStream_t g_transfer_stream = nullptr;

KvTransferResult ErrorResult(std::string error) {
  return {.success = false,
          .completion_event_handle = {},
          .error = std::move(error),
          .stats = {}};
}

std::string CudaUuidToString(const cudaUUID_t& uuid) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(&uuid);
  char out[41];
  std::snprintf(out, sizeof(out),
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-"
                "%02x%02x%02x%02x%02x%02x",
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5],
                bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11],
                bytes[12], bytes[13], bytes[14], bytes[15]);
  return out;
}

bool IpcHandleIsCached(const KvTensorMetadata& metadata) {
  const std::string key = IpcHandleCacheKey(metadata);
  std::lock_guard<std::mutex> lock(g_ipc_handles_mu);
  return g_ipc_handles.find(key) != g_ipc_handles.end();
}

bool OpenCachedIpcHandle(const KvTensorMetadata& metadata, void** ptr,
                         std::string* error) {
  const std::string key = IpcHandleCacheKey(metadata);
  {
    std::lock_guard<std::mutex> lock(g_ipc_handles_mu);
    auto it = g_ipc_handles.find(key);
    if (it != g_ipc_handles.end()) {
      *ptr = it->second;
      return true;
    }
  }

  cudaIpcMemHandle_t handle{};
  const std::size_t handle_offset =
      metadata.ipc_handle.size() - sizeof(cudaIpcMemHandle_t);
  std::memcpy(&handle, metadata.ipc_handle.data() + handle_offset,
              sizeof(handle));
  void* opened = nullptr;
  if (!CheckCuda(
          cudaIpcOpenMemHandle(&opened, handle, cudaIpcMemLazyEnablePeerAccess),
          "cudaIpcOpenMemHandle", error)) {
    return false;
  }

  std::lock_guard<std::mutex> lock(g_ipc_handles_mu);
  auto [it, inserted] = g_ipc_handles.emplace(key, opened);
  if (!inserted) {
    (void)cudaIpcCloseMemHandle(opened);
    opened = it->second;
  }
  *ptr = opened;
  return true;
}

bool SynchronizeTransferStream(std::string* error) {
  std::string local_error;
  std::string* out_error = error == nullptr ? &local_error : error;
  std::lock_guard<std::mutex> lock(g_transfer_stream_mu);
  if (g_transfer_stream == nullptr) {
    return true;
  }
  return CheckCuda(cudaStreamSynchronize(g_transfer_stream),
                   "cudaStreamSynchronize native transfer", out_error);
}

void DestroyCompletionEvents() {
  std::vector<cudaEvent_t> events;
  {
    std::lock_guard<std::mutex> lock(g_events_mu);
    events.swap(g_completion_events);
  }
  for (cudaEvent_t event : events) {
    (void)cudaEventDestroy(event);
  }
}

bool ClearDeviceChunks(std::string* error) {
  // Hot-cache STORE/RETRIEVE can return after queuing work on this stream.
  // Drain it before freeing any device chunk that queued work may reference.
  if (!SynchronizeTransferStream(error)) {
    return false;
  }
  std::lock_guard<std::mutex> lock(g_device_chunks_mu);
  for (auto& entry : g_device_chunks) {
    (void)cudaFree(entry.second.ptr);
  }
  g_device_chunks.clear();
  DestroyCompletionEvents();
  return true;
}

#endif  // LMCACHE_ENABLE_CUDA

}  // namespace

std::mutex& CudaTransferMutex() {
  static std::mutex mu;
  return mu;
}

#if LMCACHE_ENABLE_CUDA

std::string CudaErrorString(cudaError_t err, const std::string& op) {
  std::ostringstream out;
  out << op << " failed: " << cudaGetErrorName(err) << " ("
      << cudaGetErrorString(err) << ")";
  return out.str();
}

bool CheckCuda(cudaError_t err, const std::string& op, std::string* error) {
  if (err == cudaSuccess) {
    return true;
  }
  *error = CudaErrorString(err, op);
  return false;
}

bool WaitForCudaInputEvent(const std::vector<std::uint8_t>& event_handle,
                           std::string* error) {
  if (event_handle.empty()) {
    return true;
  }
  if (event_handle.size() != sizeof(cudaIpcEventHandle_t)) {
    *error = "input CUDA event IPC handle has unexpected size";
    return false;
  }
  cudaIpcEventHandle_t handle{};
  std::memcpy(&handle, event_handle.data(), sizeof(handle));
  cudaEvent_t event{};
  if (!CheckCuda(cudaIpcOpenEventHandle(&event, handle),
                 "cudaIpcOpenEventHandle", error)) {
    return false;
  }
  const bool ok =
      CheckCuda(cudaEventSynchronize(event), "cudaEventSynchronize", error);
  (void)cudaEventDestroy(event);
  return ok;
}

bool WaitForCudaInputEventOnStream(
    const std::vector<std::uint8_t>& event_handle, void* cuda_stream,
    std::string* error) {
  if (event_handle.empty()) {
    return true;
  }
  if (event_handle.size() != sizeof(cudaIpcEventHandle_t)) {
    *error = "input CUDA event IPC handle has unexpected size";
    return false;
  }
  cudaIpcEventHandle_t handle{};
  std::memcpy(&handle, event_handle.data(), sizeof(handle));
  cudaEvent_t event{};
  if (!CheckCuda(cudaIpcOpenEventHandle(&event, handle),
                 "cudaIpcOpenEventHandle", error)) {
    return false;
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const bool ok =
      CheckCuda(cudaStreamWaitEvent(stream, event, 0), "cudaStreamWaitEvent",
                error);
  (void)cudaEventDestroy(event);
  return ok;
}

bool OpenCudaTensorMemory(const KvTensorMetadata& metadata, void** ptr,
                          std::string* error) {
  if (metadata.ipc_handle.size() < sizeof(cudaIpcMemHandle_t)) {
    *error = "CUDA memory IPC handle has unexpected size";
    return false;
  }
  BestEffortSetCudaDeviceForUuid(metadata.device_uuid);
  const bool handle_cached = IpcHandleIsCached(metadata);
  if (!handle_cached && metadata.event_sync_required &&
      !WaitForCudaInputEvent(metadata.event_handle, error)) {
    return false;
  }
  return OpenCachedIpcHandle(metadata, ptr, error);
}

void BestEffortSetCudaDeviceForUuid(const std::string& device_uuid) {
  if (device_uuid.empty()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(g_device_uuid_mu);
    if (g_current_device_uuid == device_uuid) {
      return;
    }
    auto it = g_device_uuid_to_index.find(device_uuid);
    if (it != g_device_uuid_to_index.end()) {
      if (cudaSetDevice(it->second) == cudaSuccess) {
        g_current_device_uuid = device_uuid;
      }
      return;
    }
  }
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return;
  }
  for (int device = 0; device < count; ++device) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
      continue;
    }
    const std::string cuda_uuid = CudaUuidToString(prop.uuid);
    const std::string bare_cuda_uuid =
        cuda_uuid.rfind("GPU-", 0) == 0 ? cuda_uuid.substr(4) : cuda_uuid;
    if (cuda_uuid == device_uuid || bare_cuda_uuid == device_uuid) {
      if (cudaSetDevice(device) == cudaSuccess) {
        std::lock_guard<std::mutex> lock(g_device_uuid_mu);
        g_device_uuid_to_index[device_uuid] = device;
        g_device_uuid_to_index[cuda_uuid] = device;
        g_device_uuid_to_index[bare_cuda_uuid] = device;
        g_current_device_uuid = device_uuid;
      }
      return;
    }
  }
}

void* CudaTransferStream(std::string* error) {
  std::lock_guard<std::mutex> lock(g_transfer_stream_mu);
  if (g_transfer_stream == nullptr &&
      !CheckCuda(cudaStreamCreateWithFlags(&g_transfer_stream,
                                           cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags native transfer", error)) {
    return nullptr;
  }
  return g_transfer_stream;
}

bool EnsureCudaDeviceChunk(const std::string& key, std::uint64_t size,
                           void** ptr, std::string* error) {
  {
    std::lock_guard<std::mutex> lock(g_device_chunks_mu);
    auto it = g_device_chunks.find(key);
    if (it != g_device_chunks.end() && it->second.size == size) {
      it->second.ready = false;
      it->second.ready_event = nullptr;
      *ptr = it->second.ptr;
      return true;
    }
  }

  void* allocated = nullptr;
  cudaError_t err = cudaMalloc(&allocated, size);
  if (err != cudaSuccess) {
    *error = CudaErrorString(err, "cudaMalloc device chunk");
    return false;
  }

  std::lock_guard<std::mutex> lock(g_device_chunks_mu);
  auto it = g_device_chunks.find(key);
  if (it != g_device_chunks.end()) {
    (void)cudaFree(it->second.ptr);
    it->second = {.ptr = allocated,
                  .size = size,
                  .ready = false,
                  .ready_event = nullptr};
  } else {
    g_device_chunks.emplace(key, DeviceChunk{.ptr = allocated,
                                             .size = size,
                                             .ready = false,
                                             .ready_event = nullptr});
  }
  *ptr = allocated;
  return true;
}

bool FindCudaDeviceChunk(const std::string& key, std::uint64_t size,
                         void** ptr) {
  std::lock_guard<std::mutex> lock(g_device_chunks_mu);
  auto it = g_device_chunks.find(key);
  if (it == g_device_chunks.end() || it->second.size != size) {
    return false;
  }
  *ptr = it->second.ptr;
  return true;
}

void EraseCudaDeviceChunks(const std::vector<std::string>& keys) {
  std::lock_guard<std::mutex> lock(g_device_chunks_mu);
  for (const std::string& key : keys) {
    auto it = g_device_chunks.find(key);
    if (it == g_device_chunks.end()) {
      continue;
    }
    (void)cudaFree(it->second.ptr);
    g_device_chunks.erase(it);
  }
}

bool MarkCudaDeviceChunksReadyOnStream(const std::vector<std::string>& keys,
                                       void* cuda_stream, std::string* error) {
  cudaEvent_t event{};
  if (!CheckCuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                 "cudaEventCreateWithFlags device chunk ready", error)) {
    return false;
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (!CheckCuda(cudaEventRecord(event, stream),
                 "cudaEventRecord device chunk ready", error)) {
    (void)cudaEventDestroy(event);
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(g_device_chunks_mu);
    for (const std::string& key : keys) {
      auto it = g_device_chunks.find(key);
      if (it == g_device_chunks.end()) {
        continue;
      }
      it->second.ready = false;
      it->second.ready_event = event;
    }
  }
  std::lock_guard<std::mutex> lock(g_events_mu);
  g_completion_events.push_back(event);
  return true;
}

KvTransferResult MakeCudaCompletionEvent() {
  return MakeCudaCompletionEventOnStream(nullptr);
}

KvTransferResult MakeCudaCompletionEventOnStream(void* cuda_stream) {
  std::string error;
  cudaEvent_t event{};
  if (!CheckCuda(cudaEventCreateWithFlags(
                     &event, cudaEventDisableTiming | cudaEventInterprocess),
                 "cudaEventCreateWithFlags", &error)) {
    return ErrorResult(error);
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (!CheckCuda(cudaEventRecord(event, stream), "cudaEventRecord", &error)) {
    (void)cudaEventDestroy(event);
    return ErrorResult(error);
  }
  cudaIpcEventHandle_t handle{};
  if (!CheckCuda(cudaIpcGetEventHandle(&handle, event), "cudaIpcGetEventHandle",
                 &error)) {
    (void)cudaEventDestroy(event);
    return ErrorResult(error);
  }

  std::vector<std::uint8_t> handle_bytes(sizeof(handle));
  std::memcpy(handle_bytes.data(), &handle, sizeof(handle));
  // The importing process can open and wait on the IPC handle after the local
  // event object is destroyed, as long as this process keeps its CUDA context
  // alive. Do not retain one event per transfer; long AIPerf runs can otherwise
  // exhaust CUDA event resources.
  (void)cudaEventDestroy(event);
  return {.success = true,
          .completion_event_handle = std::move(handle_bytes),
          .error = {}};
}

#endif  // LMCACHE_ENABLE_CUDA

CudaTransferDeviceCacheStats GetCudaRuntimeDeviceCacheStats() {
#if LMCACHE_ENABLE_CUDA
  std::lock_guard<std::mutex> lock(g_device_chunks_mu);
  CudaTransferDeviceCacheStats stats;
  stats.entries = g_device_chunks.size();
  for (const auto& entry : g_device_chunks) {
    stats.bytes += entry.second.size;
  }
  return stats;
#else
  return {};
#endif
}

bool CudaDeviceChunkReady(const std::string& key) {
#if LMCACHE_ENABLE_CUDA
  std::lock_guard<std::mutex> lock(g_device_chunks_mu);
  auto it = g_device_chunks.find(key);
  if (it == g_device_chunks.end()) {
    return false;
  }
  if (it->second.ready) {
    return true;
  }
  if (it->second.ready_event == nullptr) {
    return false;
  }
  const cudaError_t err = cudaEventQuery(it->second.ready_event);
  if (err == cudaSuccess) {
    it->second.ready = true;
    it->second.ready_event = nullptr;
    return true;
  }
  return false;
#else
  (void)key;
  return false;
#endif
}

bool ClearCudaDeviceChunks(std::string* error) {
#if LMCACHE_ENABLE_CUDA
  return ClearDeviceChunks(error);
#else
  if (error != nullptr) {
    error->clear();
  }
  return true;
#endif
}

void ReleaseCudaRuntimeState() {
#if LMCACHE_ENABLE_CUDA
  std::string ignored_error;
  (void)ClearDeviceChunks(&ignored_error);
  {
    std::lock_guard<std::mutex> lock(g_ipc_handles_mu);
    for (auto& entry : g_ipc_handles) {
      (void)cudaIpcCloseMemHandle(entry.second);
    }
    g_ipc_handles.clear();
  }
  {
    std::lock_guard<std::mutex> lock(g_device_uuid_mu);
    g_device_uuid_to_index.clear();
    g_current_device_uuid.clear();
  }
  {
    std::lock_guard<std::mutex> lock(g_transfer_stream_mu);
    if (g_transfer_stream != nullptr) {
      (void)cudaStreamDestroy(g_transfer_stream);
      g_transfer_stream = nullptr;
    }
  }
  std::lock_guard<std::mutex> lock(g_events_mu);
  for (cudaEvent_t event : g_completion_events) {
    (void)cudaEventDestroy(event);
  }
  g_completion_events.clear();
#endif
}

}  // namespace lmcache::mp
