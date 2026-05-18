// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "lmcache_mp_cpp/cuda_transfer.h"
#include "lmcache_mp_cpp/http_server.h"
#include "lmcache_mp_cpp/ipc_key.h"
#include "lmcache_mp_cpp/l2_adapter.h"
#include "lmcache_mp_cpp/msgpack_lite.h"
#include "lmcache_mp_cpp/tiered_cache.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lmcache::mp {

struct NativeServerConfig {
  std::string host = "localhost";
  std::uint16_t port = 5555;
  std::string http_host = "0.0.0.0";
  std::uint16_t http_port = 8080;
  std::uint64_t dram_capacity_bytes = 0;
  std::string disk_path = "/tmp/lmcache-mp-native";
  std::uint32_t chunk_size = 256;
  std::uint32_t max_workers = 1;
  std::size_t max_queued_tasks = 1024;
  std::string eviction_policy = "LRU";
  std::string startup_log_level;
  std::string lmcache_version;
  std::string lmcache_commit_id;
  std::vector<L2AdapterConfig> l2_adapter_configs;
  bool enable_http = true;
  bool enable_cuda_gpu_hot_cache = false;
};

struct NativeServerMetrics {
  std::atomic<std::uint64_t> request_count{0};
  std::atomic<std::uint64_t> unsupported_count{0};
  std::atomic<std::uint64_t> clear_count{0};
  std::atomic<std::uint64_t> store_count{0};
  std::atomic<std::uint64_t> retrieve_count{0};
  std::atomic<std::uint64_t> lookup_count{0};
  std::atomic<std::uint64_t> lookup_result_fast_path_count{0};
  std::atomic<std::uint64_t> invalid_payload_count{0};
  std::atomic<std::uint64_t> block_allocation_report_count{0};
  std::atomic<std::uint64_t> block_allocation_record_count{0};
  std::atomic<std::uint64_t> cache_hits{0};
  std::atomic<std::uint64_t> cache_misses{0};
  std::atomic<std::uint64_t> partial_hit_count{0};
  std::atomic<std::uint64_t> l1_hit_count{0};
  std::atomic<std::uint64_t> l2_hit_count{0};
  std::atomic<std::uint64_t> l2_miss_count{0};
  std::atomic<std::uint64_t> l2_store_count{0};
  std::atomic<std::uint64_t> l2_load_count{0};
  std::atomic<std::uint64_t> l2_error_count{0};
  std::atomic<std::uint64_t> transfer_lock_count{0};
  std::atomic<std::uint64_t> transfer_lock_failure_count{0};
  std::atomic<std::uint64_t> transfer_lock_wait_total_us{0};
  std::atomic<std::uint64_t> transfer_lock_wait_max_us{0};
  std::atomic<std::uint64_t> transfer_lock_hold_total_us{0};
  std::atomic<std::uint64_t> transfer_lock_hold_max_us{0};
  std::atomic<std::uint64_t> queue_full_count{0};
  std::atomic<std::uint64_t> active_worker_count{0};
  std::atomic<std::uint64_t> request_queue_wait_count{0};
  std::atomic<std::uint64_t> request_queue_wait_total_us{0};
  std::atomic<std::uint64_t> request_queue_wait_max_us{0};
  std::atomic<std::uint64_t> lookup_request_queue_wait_count{0};
  std::atomic<std::uint64_t> lookup_request_queue_wait_total_us{0};
  std::atomic<std::uint64_t> lookup_request_queue_wait_max_us{0};
  std::atomic<std::uint64_t> store_request_queue_wait_count{0};
  std::atomic<std::uint64_t> store_request_queue_wait_total_us{0};
  std::atomic<std::uint64_t> store_request_queue_wait_max_us{0};
  std::atomic<std::uint64_t> retrieve_request_queue_wait_count{0};
  std::atomic<std::uint64_t> retrieve_request_queue_wait_total_us{0};
  std::atomic<std::uint64_t> retrieve_request_queue_wait_max_us{0};
  std::atomic<std::uint64_t> free_lookup_locks_request_queue_wait_count{0};
  std::atomic<std::uint64_t> free_lookup_locks_request_queue_wait_total_us{0};
  std::atomic<std::uint64_t> free_lookup_locks_request_queue_wait_max_us{0};
  std::atomic<std::uint64_t> request_latency_count{0};
  std::atomic<std::uint64_t> request_latency_total_us{0};
  std::atomic<std::uint64_t> request_latency_max_us{0};
  std::atomic<std::uint64_t> lookup_request_latency_count{0};
  std::atomic<std::uint64_t> lookup_request_latency_total_us{0};
  std::atomic<std::uint64_t> lookup_request_latency_max_us{0};
  std::atomic<std::uint64_t> store_request_latency_count{0};
  std::atomic<std::uint64_t> store_request_latency_total_us{0};
  std::atomic<std::uint64_t> store_request_latency_max_us{0};
  std::atomic<std::uint64_t> retrieve_request_latency_count{0};
  std::atomic<std::uint64_t> retrieve_request_latency_total_us{0};
  std::atomic<std::uint64_t> retrieve_request_latency_max_us{0};
  std::atomic<std::uint64_t> free_lookup_locks_request_latency_count{0};
  std::atomic<std::uint64_t> free_lookup_locks_request_latency_total_us{0};
  std::atomic<std::uint64_t> free_lookup_locks_request_latency_max_us{0};
  std::atomic<std::uint64_t> request_latency_le_100us{0};
  std::atomic<std::uint64_t> request_latency_le_500us{0};
  std::atomic<std::uint64_t> request_latency_le_1ms{0};
  std::atomic<std::uint64_t> request_latency_le_5ms{0};
  std::atomic<std::uint64_t> request_latency_le_10ms{0};
  std::atomic<std::uint64_t> request_latency_le_50ms{0};
  std::atomic<std::uint64_t> request_latency_le_100ms{0};
  std::atomic<std::uint64_t> request_latency_gt_100ms{0};
  std::atomic<std::uint64_t> cuda_transfer_store_total_us{0};
  std::atomic<std::uint64_t> cuda_transfer_store_max_us{0};
  std::atomic<std::uint64_t> cuda_transfer_retrieve_total_us{0};
  std::atomic<std::uint64_t> cuda_transfer_retrieve_max_us{0};
  std::atomic<std::uint64_t> cuda_transfer_wait_event_total_us{0};
  std::atomic<std::uint64_t> cuda_transfer_open_tensors_total_us{0};
  std::atomic<std::uint64_t> cuda_transfer_copy_total_us{0};
  std::atomic<std::uint64_t> cuda_transfer_cache_total_us{0};
  std::atomic<std::uint64_t> cuda_transfer_completion_event_total_us{0};
  std::atomic<std::uint64_t> cuda_transfer_bytes{0};
  std::atomic<std::uint64_t> cuda_transfer_memcpy_calls{0};
  std::atomic<std::uint64_t> cuda_transfer_kernel_calls{0};
};

class NativeServer {
 public:
  using Frame = std::vector<std::uint8_t>;
  using Frames = std::vector<Frame>;

  explicit NativeServer(NativeServerConfig config);
  ~NativeServer();

  NativeServer(const NativeServer&) = delete;
  NativeServer& operator=(const NativeServer&) = delete;

  bool Start();
  void Stop();
  void Wait();

 private:
  struct RequestTask {
    Frames prefix;
    std::uint8_t request_type = 0;
    Frames payloads;
    std::chrono::steady_clock::time_point enqueued_at;
  };

  struct RegisteredContext {
    std::string model_name;
    std::uint32_t world_size = 0;
    std::string engine_type;
    std::string kv_layout;
    std::uint32_t inference_engine_logical_block_size = 0;
    bool use_layerwise_hint = false;
    bool trt_llm_layout_hints = false;
    std::uint64_t trt_llm_num_kv_heads = 0;
    std::uint64_t trt_llm_tokens_per_block = 0;
    std::uint64_t trt_llm_head_dim = 0;
    std::uint64_t kv_cache_wrapper_count = 0;
    std::string first_kv_dtype;
    std::vector<std::uint64_t> first_kv_shape;
    std::vector<std::uint64_t> first_kv_stride;
    std::string first_kv_device_uuid;
    std::uint64_t first_kv_num_blocks = 0;
    std::uint64_t first_kv_block_size = 0;
    std::uint64_t first_kv_ipc_handle_bytes = 0;
    std::uint64_t first_kv_storage_bytes = 0;
    std::uint64_t first_kv_storage_offset_bytes = 0;
    std::uint64_t first_kv_event_handle_bytes = 0;
    bool first_kv_event_sync_required = false;
    std::vector<msgpack::DecodedCudaIpcWrapper> kv_wrappers;
  };

  struct BlockAllocationSummary {
    bool valid = false;
    std::uint64_t instance_id = 0;
    std::string model_name;
    std::uint64_t record_count = 0;
    std::string last_request_id;
    std::uint64_t last_new_block_count = 0;
    std::uint64_t last_new_token_count = 0;
  };

  struct KvTransferPlan {
    IpcCacheEngineKey key;
    RegisteredContext context;
    std::uint64_t instance_id = 0;
    std::vector<std::uint64_t> gpu_block_ids;
    Frame event_handle;
    std::vector<std::string> object_keys;
    std::uint64_t skip_first_n_tokens = 0;
    std::uint64_t blocks_per_chunk = 0;
  };

  void ZmqLoop();
  void WorkerLoop();
  void RecordClientIdentity(const Frame& identity);
  std::size_t ObservedClientCount() const;
  void RecordRequestLatency(std::uint8_t request_type,
                            std::uint64_t elapsed_us);
  void RecordRequestQueueWait(std::uint8_t request_type,
                              std::uint64_t elapsed_us);
  void RecordCudaTransferStats(bool retrieve, const KvTransferStats& stats);
  void EnqueueTask(RequestTask task);
  void EnqueueResponse(Frames frames);
  bool TryHandleImmediateRequest(const RequestTask& task, Frames* response);
  bool TryHandleReadyLookupResult(const RequestTask& task, Frames* response);
  bool DrainResponses(void* socket);
  Frames HandleRequest(const RequestTask& task);
  void LaunchCudaTensorWarmup(std::vector<KvTensorMetadata> tensors);
  void JoinBackgroundTasks();
  HttpResponse HandleHttp(const std::string& method, const std::string& path,
                          const std::string& body);
  HttpResponse HandleLogLevelHttp(const std::string& path);
  HttpResponse HandleThreadsHttp(const std::string& path);
  HttpResponse HandlePeriodicThreadsHttp(const std::string& path);
  HttpResponse HandleQuotaHttp(const std::string& method,
                               const std::string& path,
                               const std::string& body);
  HttpResponse HandleKvCacheCheckHttp(const std::string& path);
  std::string StatusJson() const;
  std::string ConfigJson() const;
  std::string PrometheusMetricsText() const;
  void ResetMetrics();
  void ForceClearCache();
  void HandleRegisterKvCache(const Frames& payloads);
  void HandleUnregisterKvCache(const Frames& payloads);
  void HandleReportBlockAllocation(const Frames& payloads);
  std::uint64_t HandleLookupPayload(const Frames& payloads, bool store_result);
  bool ValidateCbRegisterKvCache(const Frames& payloads);
  bool ValidateCbUnregisterKvCache(const Frames& payloads);
  bool ValidateCbLookupPayload(const Frames& payloads);
  bool ValidateCbStorePayload(const Frames& payloads);
  bool ValidateCbRetrievePayload(const Frames& payloads, bool v2);
  void HandleFreeLookupLocks(const Frames& payloads);
  void HandleEndSession(const Frames& payloads);
  bool ValidateKvTransferPayload(const Frames& payloads, bool retrieve);
  std::optional<KvTransferPlan> BuildKvTransferPlan(const Frames& payloads,
                                                    bool retrieve);
  Frames HandleStorePayload(const RequestTask& task);
  Frames HandleRetrievePayload(const RequestTask& task);
  bool WriteStoredChunksToL2(const std::vector<std::string>& object_keys);
  void LoadMissingChunksFromL2(const std::vector<std::string>& object_keys);
  void ClearL2Adapters();
  bool LockTransferChunks(const std::vector<std::string>& object_keys,
                          std::vector<std::string>* locked_object_keys,
                          std::uint64_t* lock_epoch,
                          const std::string& context);
  bool RecordInvalidPayload(const std::string& message);
  void ReleaseLocks(const std::vector<std::string>& object_keys);
  void ReleaseLocksForEpoch(const std::vector<std::string>& object_keys,
                            std::uint64_t lock_epoch);
  void ReleaseLocksForRequest(const std::string& request_id);
  void ReleaseLocksForRequestKeys(const std::string& request_id,
                                  const std::vector<std::string>& object_keys);
  std::uint64_t LookupResult(const Frames& payloads, bool erase_result);
  bool HasRegisteredContext(const IpcCacheEngineKey& key) const;
  std::size_t RegisteredContextCount() const;
  std::string RegisteredContextsJson() const;
  std::string LastBlockAllocationJson() const;

  NativeServerConfig config_;
  NativeServerMetrics metrics_;
  LmcacheMpCppCache* cache_ = nullptr;
  std::atomic<std::uint64_t> force_clear_epoch_{0};
  void* zmq_context_ = nullptr;
  std::atomic<bool> stop_{false};
  std::thread zmq_thread_;
  std::vector<std::thread> workers_;
  std::unique_ptr<HttpServer> http_server_;
  std::vector<std::unique_ptr<L2Adapter>> l2_adapters_;
  mutable std::mutex l2_adapters_mu_;

  mutable std::mutex tasks_mu_;
  std::condition_variable tasks_cv_;
  std::deque<RequestTask> tasks_;

  mutable std::mutex responses_mu_;
  std::deque<Frames> responses_;

  mutable std::mutex clients_mu_;
  std::unordered_map<std::string, std::uint64_t> client_request_counts_;

  mutable std::mutex log_levels_mu_;
  std::unordered_map<std::string, std::string> log_levels_;

  mutable std::mutex quota_mu_;
  std::unordered_map<std::string, std::uint64_t> quota_limits_;

  mutable std::mutex lookup_results_mu_;
  std::unordered_map<std::string, std::uint64_t> lookup_results_;
  std::unordered_map<std::string, std::vector<std::string>> lookup_locks_;

  mutable std::mutex registered_contexts_mu_;
  std::unordered_map<std::uint64_t, RegisteredContext> registered_contexts_;

  mutable std::mutex block_allocations_mu_;
  BlockAllocationSummary last_block_allocation_;

  mutable std::mutex background_tasks_mu_;
  std::vector<std::thread> background_tasks_;
};

}  // namespace lmcache::mp
