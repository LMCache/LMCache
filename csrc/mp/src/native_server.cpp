// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/native_server.h"

#include "lmcache_mp_cpp/msgpack_lite.h"
#include "lmcache_mp_cpp/native_http_utils.h"
#include "lmcache_mp_cpp/protocol.h"

#include <zmq.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>

namespace lmcache::mp {
namespace {

constexpr std::size_t kMaxZmqFrameBytes = 16 * 1024 * 1024;

enum class TaskLane {
  kControl,
  kRetrieve,
  kLookup,
  kStore,
};

TaskLane TaskLaneForRequest(std::uint8_t request_type) {
  switch (static_cast<RequestType>(request_type)) {
    case RequestType::kQueryPrefetchStatus:
    case RequestType::kQueryPrefetchLookupHits:
    case RequestType::kPing:
    case RequestType::kGetChunkSize:
    case RequestType::kNoop:
    case RequestType::kRegisterKvCache:
    case RequestType::kUnregisterKvCache:
    case RequestType::kReportBlockAllocation:
      return TaskLane::kControl;
    case RequestType::kRetrieve:
    case RequestType::kCbRetrievePreComputed:
    case RequestType::kCbRetrievePreComputedV2:
      return TaskLane::kRetrieve;
    case RequestType::kLookup:
    case RequestType::kFreeLookupLocks:
    case RequestType::kEndSession:
    case RequestType::kCbLookupPreComputed:
    case RequestType::kCbLookupPreComputedV2:
      return TaskLane::kLookup;
    case RequestType::kStore:
    case RequestType::kCbStorePreComputed:
    case RequestType::kCbStoreFinal:
      return TaskLane::kStore;
    default:
      return TaskLane::kControl;
  }
}

std::string FrameToString(const NativeServer::Frame& frame) {
  return std::string(frame.begin(), frame.end());
}

bool RecvFrame(void* socket, NativeServer::Frame* frame, bool* more) {
  zmq_msg_t msg;
  if (zmq_msg_init(&msg) != 0) {
    return false;
  }
  const int rc = zmq_msg_recv(&msg, socket, 0);
  if (rc < 0) {
    zmq_msg_close(&msg);
    return false;
  }
  const auto* data = static_cast<const std::uint8_t*>(zmq_msg_data(&msg));
  frame->assign(data, data + zmq_msg_size(&msg));
  int more_value = 0;
  std::size_t more_size = sizeof(more_value);
  zmq_getsockopt(socket, ZMQ_RCVMORE, &more_value, &more_size);
  *more = more_value != 0;
  zmq_msg_close(&msg);
  return true;
}

bool SendFrame(void* socket, const NativeServer::Frame& frame, bool more) {
  zmq_msg_t msg;
  if (zmq_msg_init_size(&msg, frame.size()) != 0) {
    return false;
  }
  if (!frame.empty()) {
    std::memcpy(zmq_msg_data(&msg), frame.data(), frame.size());
  }
  const int flags = more ? ZMQ_SNDMORE : 0;
  const int rc = zmq_msg_send(&msg, socket, flags);
  zmq_msg_close(&msg);
  return rc >= 0;
}

bool SendMultipart(void* socket, const NativeServer::Frames& frames) {
  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (!SendFrame(socket, frames[i], i + 1 < frames.size())) {
      return false;
    }
  }
  return true;
}

void AtomicMax(std::atomic<std::uint64_t>* value, std::uint64_t candidate) {
  std::uint64_t current = value->load(std::memory_order_relaxed);
  while (current < candidate &&
         !value->compare_exchange_weak(current, candidate,
                                       std::memory_order_relaxed)) {
  }
}

void RecordLatencySummary(std::atomic<std::uint64_t>* count,
                          std::atomic<std::uint64_t>* total_us,
                          std::atomic<std::uint64_t>* max_us,
                          std::uint64_t elapsed_us) {
  count->fetch_add(1, std::memory_order_relaxed);
  total_us->fetch_add(elapsed_us, std::memory_order_relaxed);
  AtomicMax(max_us, elapsed_us);
}

std::uint64_t ElapsedMicros(std::chrono::steady_clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

}  // namespace

NativeServer::NativeServer(NativeServerConfig config)
    : config_(std::move(config)) {}

NativeServer::~NativeServer() { Stop(); }

bool NativeServer::Start() {
  if (config_.chunk_size == 0) {
    std::cerr << "native MP server requires positive --chunk-size\n";
    return false;
  }
  if (config_.eviction_policy != "LRU") {
    std::cerr << "native MP server only supports --eviction-policy LRU; got "
              << config_.eviction_policy << "\n";
    return false;
  }
  if (!config_.startup_log_level.empty()) {
    const std::string canonical_level =
        UppercaseAscii(config_.startup_log_level);
    if (!IsSupportedLogLevel(canonical_level)) {
      std::cerr << "invalid native MP --log-level: "
                << config_.startup_log_level << "\n";
      return false;
    }
    config_.startup_log_level = canonical_level;
    std::lock_guard<std::mutex> lock(log_levels_mu_);
    log_levels_["lmcache"] = canonical_level;
    log_levels_["lmcache.native"] = canonical_level;
  }
  cache_ = lmcache_mp_cpp_cache_create(config_.dram_capacity_bytes,
                                       config_.disk_path.c_str());
  if (cache_ == nullptr) {
    std::cerr << "failed to create native tiered cache\n";
    return false;
  }
  for (const L2AdapterConfig& adapter_config : config_.l2_adapter_configs) {
    std::string error;
    auto adapter = CreateL2Adapter(adapter_config, &error);
    if (!adapter) {
      std::cerr << "failed to create native L2 adapter: " << error << "\n";
      return false;
    }
    l2_adapters_.push_back(std::move(adapter));
  }

  stop_.store(false);
  zmq_context_ = zmq_ctx_new();
  if (zmq_context_ == nullptr) {
    std::cerr << "failed to create ZMQ context\n";
    return false;
  }

  const std::uint32_t workers =
      config_.max_workers == 0 ? 1 : config_.max_workers;
  for (std::uint32_t i = 0; i < workers; ++i) {
    workers_.emplace_back(&NativeServer::WorkerLoop, this);
  }
  zmq_thread_ = std::thread(&NativeServer::ZmqLoop, this);

  if (config_.enable_http) {
    http_server_ = std::make_unique<HttpServer>(
        config_.http_host, config_.http_port,
        [this](const std::string& method, const std::string& path,
               const std::string& body) {
          return HandleHttp(method, path, body);
        });
    if (!http_server_->Start()) {
      Stop();
      return false;
    }
  }

  std::cerr << "LMCache native MP server listening on tcp://" << config_.host
            << ":" << config_.port << ", HTTP " << config_.http_host << ":"
            << config_.http_port << "\n";
  return true;
}

void NativeServer::Stop() {
  const bool was_stopped = stop_.exchange(true);
  if (!was_stopped) {
    tasks_cv_.notify_all();
  }
  if (http_server_) {
    http_server_->Stop();
    http_server_.reset();
  }
  if (zmq_context_ != nullptr) {
    zmq_ctx_shutdown(zmq_context_);
  }
  if (zmq_thread_.joinable()) {
    zmq_thread_.join();
  }
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();
  JoinBackgroundTasks();
  if (zmq_context_ != nullptr) {
    zmq_ctx_term(zmq_context_);
    zmq_context_ = nullptr;
  }
  if (cache_ != nullptr) {
    lmcache_mp_cpp_cache_destroy(cache_);
    cache_ = nullptr;
  }
  ReleaseCudaTransferEvents();
  l2_adapters_.clear();
}

void NativeServer::Wait() {
  if (zmq_thread_.joinable()) {
    zmq_thread_.join();
  }
}

void NativeServer::LaunchCudaTensorWarmup(
    std::vector<KvTensorMetadata> tensors) {
  if (!config_.enable_cuda_gpu_hot_cache || tensors.empty() ||
      stop_.load(std::memory_order_acquire)) {
    return;
  }

  std::lock_guard<std::mutex> lock(background_tasks_mu_);
  if (stop_.load(std::memory_order_acquire)) {
    return;
  }
  background_tasks_.emplace_back([tensors = std::move(tensors)]() {
    std::string error;
    if (!WarmCudaTransferTensorHandles(tensors, &error)) {
      std::cerr << "native CUDA tensor warmup failed: " << error << "\n";
    }
  });
}

void NativeServer::JoinBackgroundTasks() {
  std::vector<std::thread> tasks;
  {
    std::lock_guard<std::mutex> lock(background_tasks_mu_);
    tasks.swap(background_tasks_);
  }
  for (std::thread& task : tasks) {
    if (task.joinable()) {
      task.join();
    }
  }
}

void NativeServer::ZmqLoop() {
  void* socket = zmq_socket(zmq_context_, ZMQ_ROUTER);
  if (socket == nullptr) {
    std::cerr << "failed to create ZMQ ROUTER socket\n";
    Stop();
    return;
  }
  const int linger_ms = 0;
  (void)zmq_setsockopt(socket, ZMQ_LINGER, &linger_ms, sizeof(linger_ms));

  const std::string bind_url =
      "tcp://" + config_.host + ":" + std::to_string(config_.port);
  if (zmq_bind(socket, bind_url.c_str()) != 0) {
    std::cerr << "failed to bind " << bind_url << ": " << zmq_strerror(errno)
              << "\n";
    zmq_close(socket);
    Stop();
    return;
  }

  while (!stop_.load()) {
    bool sent_direct_response = false;
    zmq_pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
    const int rc = zmq_poll(items, 1, 1);
    if (rc < 0) {
      if (!stop_.load()) {
        std::cerr << "ZMQ poll failed: " << zmq_strerror(errno) << "\n";
      }
      break;
    }
    if ((items[0].revents & ZMQ_POLLIN) != 0) {
      Frames frames;
      bool more = false;
      bool oversized_frame = false;
      do {
        Frame frame;
        if (!RecvFrame(socket, &frame, &more)) {
          break;
        }
        oversized_frame = oversized_frame || frame.size() > kMaxZmqFrameBytes;
        frames.push_back(std::move(frame));
      } while (more);

      if (frames.empty()) {
        continue;
      }

      if (oversized_frame) {
        metrics_.invalid_payload_count.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "discarding oversized ZMQ request frame; max frame bytes "
                  << kMaxZmqFrameBytes << "\n";
        if (frames.size() >= 3 && frames[0].size() <= kMaxZmqFrameBytes &&
            frames[1].size() <= kMaxZmqFrameBytes &&
            frames[2].size() <= kMaxZmqFrameBytes) {
          RecordClientIdentity(frames[0]);
          Frames response = {frames[0], frames[1], frames[2],
                             msgpack::EncodeNil()};
          EnqueueResponse(std::move(response));
          (void)DrainResponses(socket);
        }
        continue;
      }

      if (frames.size() >= 3) {
        const auto maybe_type = msgpack::DecodeUnsigned(frames[2]);
        if (!maybe_type ||
            *maybe_type > std::numeric_limits<std::uint8_t>::max()) {
          metrics_.invalid_payload_count.fetch_add(1,
                                                  std::memory_order_relaxed);
          metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
          std::cerr << "discarding malformed ZMQ request type frame\n";
          Frames response = {frames[0], frames[1], frames[2],
                             msgpack::EncodeNil()};
          EnqueueResponse(std::move(response));
          (void)DrainResponses(socket);
          continue;
        }
        const auto request_type = static_cast<RequestType>(
            static_cast<std::uint8_t>(*maybe_type));
        RecordClientIdentity(frames[0]);
        const std::size_t routing_client_count =
            ObserveRoutingClientCount(frames[0]);
        const bool route_control_through_worker =
            (request_type == RequestType::kPing ||
             request_type == RequestType::kNoop) &&
            routing_client_count > 2;
        const bool wait_for_fast_worker_response =
            route_control_through_worker;
        if (!route_control_through_worker &&
            TrySendImmediateRequest(socket, frames,
                                    static_cast<std::uint8_t>(request_type))) {
          sent_direct_response = true;
          continue;
        }

        RequestTask task;
        task.prefix = {frames[0], frames[1], frames[2]};
        task.request_type = static_cast<std::uint8_t>(request_type);
        task.payloads.assign(frames.begin() + 3, frames.end());
        Frames fast_response;
        if (TryHandleReadyLookupResult(task, &fast_response)) {
          (void)SendMultipart(socket, fast_response);
          sent_direct_response = true;
        } else {
          EnqueueTask(std::move(task));
          if (wait_for_fast_worker_response) {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::microseconds(75);
            if (WaitForResponsesUntil(socket, deadline)) {
              sent_direct_response = true;
            }
          }
        }
      } else {
        metrics_.invalid_payload_count.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "discarding malformed ZMQ request with " << frames.size()
                  << " frames\n";
      }
    }
    if (!sent_direct_response) {
      (void)DrainResponses(socket);
    }
  }

  (void)DrainResponses(socket);
  zmq_close(socket);
}

void NativeServer::RecordClientIdentity(const Frame& identity) {
  if (!config_.enable_http) {
    return;
  }
  std::lock_guard<std::mutex> lock(clients_mu_);
  ++client_request_counts_[FrameToString(identity)];
}

std::size_t NativeServer::ObservedClientCount() const {
  std::lock_guard<std::mutex> lock(clients_mu_);
  return client_request_counts_.size();
}

std::size_t NativeServer::ObserveRoutingClientCount(const Frame& identity) {
  for (const Frame& known : routing_clients_) {
    if (identity == known) {
      return routing_clients_.size();
    }
  }
  if (routing_clients_.size() < 3) {
    routing_clients_.push_back(identity);
  }
  return routing_clients_.size();
}

bool NativeServer::TrySendImmediateRequest(void* socket, const Frames& frames,
                                           std::uint8_t request_type) {
  const auto start = std::chrono::steady_clock::now();
  const bool record_metrics = config_.enable_http;
  static const Frame kTrueResponse = msgpack::EncodeBool(true);
  static const Frame kNoopResponse = msgpack::EncodeString("OK");

  const Frame* payload = nullptr;
  Frame chunk_size_payload;
  switch (static_cast<RequestType>(request_type)) {
    case RequestType::kPing:
      payload = &kTrueResponse;
      break;
    case RequestType::kNoop:
      payload = &kNoopResponse;
      break;
    case RequestType::kGetChunkSize:
      chunk_size_payload = msgpack::EncodeUnsigned(config_.chunk_size);
      payload = &chunk_size_payload;
      break;
    default:
      return false;
  }

  if (record_metrics) {
    metrics_.request_count.fetch_add(1, std::memory_order_relaxed);
    RecordRequestQueueWait(request_type, 0);
    RecordRequestLatency(request_type, ElapsedMicros(start));
  }
  return SendFrame(socket, frames[0], true) &&
         SendFrame(socket, frames[1], true) &&
         SendFrame(socket, frames[2], true) && SendFrame(socket, *payload, false);
}

void NativeServer::WorkerLoop() {
  while (true) {
    RequestTask task;
    {
      std::unique_lock<std::mutex> lock(tasks_mu_);
      tasks_cv_.wait(lock, [this] {
        return stop_.load() || HasQueuedTasksLocked();
      });
      if (stop_.load() && !HasQueuedTasksLocked()) {
        return;
      }
      task = PopTaskLocked();
    }
    const auto dequeued_at = std::chrono::steady_clock::now();
    const auto queue_wait_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            dequeued_at - task.enqueued_at)
            .count());
    RecordRequestQueueWait(task.request_type, queue_wait_us);
    metrics_.active_worker_count.fetch_add(1, std::memory_order_relaxed);
    const auto start = std::chrono::steady_clock::now();
    Frames response = HandleRequest(task);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    metrics_.active_worker_count.fetch_sub(1, std::memory_order_relaxed);
    const auto elapsed_us = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    RecordRequestLatency(task.request_type, elapsed_us);
    EnqueueResponse(std::move(response));
  }
}

void NativeServer::RecordRequestLatency(std::uint8_t request_type,
                                        std::uint64_t elapsed_us) {
  RecordLatencySummary(&metrics_.request_latency_count,
                       &metrics_.request_latency_total_us,
                       &metrics_.request_latency_max_us, elapsed_us);
  switch (static_cast<RequestType>(request_type)) {
    case RequestType::kLookup:
      RecordLatencySummary(&metrics_.lookup_request_latency_count,
                           &metrics_.lookup_request_latency_total_us,
                           &metrics_.lookup_request_latency_max_us,
                           elapsed_us);
      break;
    case RequestType::kStore:
      RecordLatencySummary(&metrics_.store_request_latency_count,
                           &metrics_.store_request_latency_total_us,
                           &metrics_.store_request_latency_max_us,
                           elapsed_us);
      break;
    case RequestType::kRetrieve:
      RecordLatencySummary(&metrics_.retrieve_request_latency_count,
                           &metrics_.retrieve_request_latency_total_us,
                           &metrics_.retrieve_request_latency_max_us,
                           elapsed_us);
      break;
    case RequestType::kFreeLookupLocks:
      RecordLatencySummary(
          &metrics_.free_lookup_locks_request_latency_count,
          &metrics_.free_lookup_locks_request_latency_total_us,
          &metrics_.free_lookup_locks_request_latency_max_us, elapsed_us);
      break;
    default:
      break;
  }
  if (elapsed_us <= 100) {
    metrics_.request_latency_le_100us.fetch_add(1, std::memory_order_relaxed);
  } else if (elapsed_us <= 500) {
    metrics_.request_latency_le_500us.fetch_add(1, std::memory_order_relaxed);
  } else if (elapsed_us <= 1000) {
    metrics_.request_latency_le_1ms.fetch_add(1, std::memory_order_relaxed);
  } else if (elapsed_us <= 5000) {
    metrics_.request_latency_le_5ms.fetch_add(1, std::memory_order_relaxed);
  } else if (elapsed_us <= 10000) {
    metrics_.request_latency_le_10ms.fetch_add(1, std::memory_order_relaxed);
  } else if (elapsed_us <= 50000) {
    metrics_.request_latency_le_50ms.fetch_add(1, std::memory_order_relaxed);
  } else if (elapsed_us <= 100000) {
    metrics_.request_latency_le_100ms.fetch_add(1, std::memory_order_relaxed);
  } else {
    metrics_.request_latency_gt_100ms.fetch_add(1, std::memory_order_relaxed);
  }
}

void NativeServer::RecordRequestQueueWait(std::uint8_t request_type,
                                          std::uint64_t elapsed_us) {
  RecordLatencySummary(&metrics_.request_queue_wait_count,
                       &metrics_.request_queue_wait_total_us,
                       &metrics_.request_queue_wait_max_us, elapsed_us);
  switch (static_cast<RequestType>(request_type)) {
    case RequestType::kLookup:
      RecordLatencySummary(&metrics_.lookup_request_queue_wait_count,
                           &metrics_.lookup_request_queue_wait_total_us,
                           &metrics_.lookup_request_queue_wait_max_us,
                           elapsed_us);
      break;
    case RequestType::kStore:
      RecordLatencySummary(&metrics_.store_request_queue_wait_count,
                           &metrics_.store_request_queue_wait_total_us,
                           &metrics_.store_request_queue_wait_max_us,
                           elapsed_us);
      break;
    case RequestType::kRetrieve:
      RecordLatencySummary(&metrics_.retrieve_request_queue_wait_count,
                           &metrics_.retrieve_request_queue_wait_total_us,
                           &metrics_.retrieve_request_queue_wait_max_us,
                           elapsed_us);
      break;
    case RequestType::kFreeLookupLocks:
      RecordLatencySummary(
          &metrics_.free_lookup_locks_request_queue_wait_count,
          &metrics_.free_lookup_locks_request_queue_wait_total_us,
          &metrics_.free_lookup_locks_request_queue_wait_max_us, elapsed_us);
      break;
    default:
      break;
  }
}

void NativeServer::RecordCudaTransferStats(bool retrieve,
                                           const KvTransferStats& stats) {
  const std::uint64_t total_us =
      stats.wait_event_us + stats.open_tensors_us + stats.copy_us +
      stats.cache_us + stats.completion_event_us;
  if (retrieve) {
    metrics_.cuda_transfer_retrieve_total_us.fetch_add(
        total_us, std::memory_order_relaxed);
    AtomicMax(&metrics_.cuda_transfer_retrieve_max_us, total_us);
  } else {
    metrics_.cuda_transfer_store_total_us.fetch_add(
        total_us, std::memory_order_relaxed);
    AtomicMax(&metrics_.cuda_transfer_store_max_us, total_us);
  }
  metrics_.cuda_transfer_wait_event_total_us.fetch_add(
      stats.wait_event_us, std::memory_order_relaxed);
  metrics_.cuda_transfer_open_tensors_total_us.fetch_add(
      stats.open_tensors_us, std::memory_order_relaxed);
  metrics_.cuda_transfer_copy_total_us.fetch_add(stats.copy_us,
                                                 std::memory_order_relaxed);
  metrics_.cuda_transfer_cache_total_us.fetch_add(stats.cache_us,
                                                  std::memory_order_relaxed);
  metrics_.cuda_transfer_completion_event_total_us.fetch_add(
      stats.completion_event_us, std::memory_order_relaxed);
  metrics_.cuda_transfer_bytes.fetch_add(stats.bytes,
                                         std::memory_order_relaxed);
  metrics_.cuda_transfer_memcpy_calls.fetch_add(stats.cuda_memcpy_calls,
                                                std::memory_order_relaxed);
  metrics_.cuda_transfer_kernel_calls.fetch_add(stats.cuda_kernel_calls,
                                                std::memory_order_relaxed);
}

void NativeServer::EnqueueTask(RequestTask task) {
  task.enqueued_at = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lock(tasks_mu_);
    if (QueuedTaskCountLocked() >= config_.max_queued_tasks) {
      metrics_.queue_full_count.fetch_add(1, std::memory_order_relaxed);
      Frames response = std::move(task.prefix);
      response.push_back(msgpack::EncodeNil());
      EnqueueResponse(std::move(response));
      return;
    }
    TrackPendingLookup(task);
    PushTaskLocked(std::move(task));
  }
  tasks_cv_.notify_one();
}

bool NativeServer::HasQueuedTasksLocked() const {
  return !control_tasks_.empty() || !retrieve_tasks_.empty() ||
         !lookup_tasks_.empty() || !store_tasks_.empty();
}

std::size_t NativeServer::QueuedTaskCountLocked() const {
  return control_tasks_.size() + retrieve_tasks_.size() + lookup_tasks_.size() +
         store_tasks_.size();
}

void NativeServer::PushTaskLocked(RequestTask task) {
  switch (TaskLaneForRequest(task.request_type)) {
    case TaskLane::kControl:
      control_tasks_.push_back(std::move(task));
      break;
    case TaskLane::kRetrieve:
      retrieve_tasks_.push_back(std::move(task));
      break;
    case TaskLane::kLookup:
      lookup_tasks_.push_back(std::move(task));
      break;
    case TaskLane::kStore:
      store_tasks_.push_back(std::move(task));
      break;
  }
}

NativeServer::RequestTask NativeServer::PopTaskLocked() {
  std::deque<RequestTask>* tasks = &store_tasks_;
  if (!control_tasks_.empty()) {
    tasks = &control_tasks_;
  } else if (!retrieve_tasks_.empty()) {
    tasks = &retrieve_tasks_;
  } else if (!lookup_tasks_.empty()) {
    tasks = &lookup_tasks_;
  }

  RequestTask task = std::move(tasks->front());
  tasks->pop_front();
  return task;
}

void NativeServer::EnqueueResponse(Frames frames) {
  {
    std::lock_guard<std::mutex> lock(responses_mu_);
    responses_.push_back(std::move(frames));
  }
  responses_cv_.notify_one();
}

bool NativeServer::TryHandleReadyLookupResult(const RequestTask& task,
                                              Frames* response) {
  const auto request_type = static_cast<RequestType>(task.request_type);
  const bool erase_result = request_type == RequestType::kQueryPrefetchStatus;
  if (!erase_result && request_type != RequestType::kQueryPrefetchLookupHits) {
    return false;
  }
  if (task.payloads.empty()) {
    return false;
  }

  const auto request_id = msgpack::DecodeString(task.payloads[0]);
  if (!request_id) {
    return false;
  }

  std::uint64_t result = 0;
  bool has_result = false;
  bool is_pending = false;
  {
    std::lock_guard<std::mutex> lock(lookup_results_mu_);
    const auto it = lookup_results_.find(*request_id);
    has_result = it != lookup_results_.end();
    if (has_result) {
      result = it->second;
    } else {
      is_pending = pending_lookup_results_.count(*request_id) != 0;
    }
    if (has_result && erase_result) {
      lookup_results_.erase(it);
    }
  }

  const auto start = std::chrono::steady_clock::now();
  metrics_.request_count.fetch_add(1, std::memory_order_relaxed);
  if (has_result) {
    metrics_.lookup_result_fast_path_count.fetch_add(1,
                                                     std::memory_order_relaxed);
  }
  RecordRequestQueueWait(task.request_type, 0);
  Frames out = task.prefix;
  out.push_back(is_pending ? msgpack::EncodeNil()
                           : msgpack::EncodeUnsigned(result));
  RecordRequestLatency(task.request_type, ElapsedMicros(start));
  *response = std::move(out);
  return true;
}

void NativeServer::TrackPendingLookup(const RequestTask& task) {
  const auto request_type = static_cast<RequestType>(task.request_type);
  if (request_type != RequestType::kLookup || task.payloads.empty()) {
    return;
  }

  std::string error;
  auto request_id = DecodeNoContextLookupRequestId(
      task.payloads[0].data(), task.payloads[0].size(), &error);
  if (!request_id) {
    return;
  }

  std::lock_guard<std::mutex> lock(lookup_results_mu_);
  pending_lookup_results_.insert(*request_id);
}

bool NativeServer::DrainResponses(void* socket) {
  std::deque<Frames> local;
  {
    std::lock_guard<std::mutex> lock(responses_mu_);
    local.swap(responses_);
  }
  if (local.empty()) {
    return false;
  }
  bool ok = true;
  for (const Frames& frames : local) {
    ok = SendMultipart(socket, frames) && ok;
  }
  return ok;
}

bool NativeServer::WaitForResponsesUntil(
    void* socket,
    std::chrono::steady_clock::time_point deadline) {
  {
    std::unique_lock<std::mutex> lock(responses_mu_);
    if (responses_.empty()) {
      responses_cv_.wait_until(lock, deadline, [this] {
        return stop_.load(std::memory_order_relaxed) || !responses_.empty();
      });
    }
  }
  return DrainResponses(socket);
}

std::size_t NativeServer::RegisteredContextCount() const {
  std::lock_guard<std::mutex> lock(registered_contexts_mu_);
  return registered_contexts_.size();
}


}  // namespace lmcache::mp
