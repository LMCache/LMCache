// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/native_server.h"

#include "lmcache_mp_cpp/native_http_utils.h"
#include "lmcache_mp_cpp/native_status.h"
#include "lmcache_mp_cpp/protocol.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

extern char** environ;

namespace lmcache::mp {
namespace {

std::uint64_t AtomicLoad(const std::atomic<std::uint64_t>& value) {
  return value.load(std::memory_order_relaxed);
}

void AtomicStoreZero(std::atomic<std::uint64_t>* value) {
  value->store(0, std::memory_order_relaxed);
}

bool StartsWith(const std::string& value, const std::string& prefix) {
  return value.rfind(prefix, 0) == 0;
}

HttpResponse UnsupportedHttpEndpoint(const std::string& path,
                                     const std::string& detail) {
  return {.status = 501,
          .content_type = "application/json",
          .body =
              "{\"error\":\"native MP HTTP endpoint not implemented\","
              "\"endpoint\":\"" +
              JsonEscape(path) + "\",\"detail\":\"" + JsonEscape(detail) +
              "\"}"};
}

std::string EnvironmentJsonText() {
  std::vector<std::pair<std::string, std::string>> entries;
  for (char** current = environ; current != nullptr && *current != nullptr;
       ++current) {
    const std::string entry(*current);
    const std::size_t delimiter = entry.find('=');
    if (delimiter == std::string::npos) {
      entries.push_back({entry, ""});
    } else {
      entries.push_back(
          {entry.substr(0, delimiter), entry.substr(delimiter + 1)});
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto& left, const auto& right) {
              return left.first < right.first;
            });

  std::ostringstream out;
  out << "{";
  for (std::size_t i = 0; i < entries.size(); ++i) {
    out << "\n  \"" << JsonEscape(entries[i].first) << "\": \""
        << JsonEscape(entries[i].second) << "\"";
    if (i + 1 < entries.size()) {
      out << ",";
    }
  }
  if (!entries.empty()) {
    out << "\n";
  }
  out << "}";
  return out.str();
}

}  // namespace

std::string NativeServer::RegisteredContextsJson() const {
  std::lock_guard<std::mutex> lock(registered_contexts_mu_);
  std::ostringstream out;
  out << "[";
  bool first = true;
  for (const auto& [instance_id, context] : registered_contexts_) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << "{"
        << "\"instance_id\":" << instance_id << ","
        << "\"model_name\":\"" << JsonEscape(context.model_name) << "\","
        << "\"world_size\":" << context.world_size << ","
        << "\"engine_type\":\"" << JsonEscape(context.engine_type) << "\","
        << "\"kv_layout\":\"" << JsonEscape(context.kv_layout) << "\","
        << "\"inference_engine_logical_block_size\":"
        << context.inference_engine_logical_block_size << ","
        << "\"use_layerwise\":"
        << (context.use_layerwise_hint ? "true" : "false") << ","
        << "\"trt_llm_layout_hints\":"
        << (context.trt_llm_layout_hints ? "true" : "false") << ","
        << "\"trt_llm_num_kv_heads\":" << context.trt_llm_num_kv_heads << ","
        << "\"trt_llm_tokens_per_block\":" << context.trt_llm_tokens_per_block
        << ","
        << "\"trt_llm_head_dim\":" << context.trt_llm_head_dim << ","
        << "\"kv_cache_wrapper_count\":" << context.kv_cache_wrapper_count
        << ","
        << "\"first_kv_dtype\":\"" << JsonEscape(context.first_kv_dtype)
        << "\","
        << "\"first_kv_shape\":" << JsonUnsignedArray(context.first_kv_shape)
        << ","
        << "\"first_kv_stride\":" << JsonUnsignedArray(context.first_kv_stride)
        << ","
        << "\"first_kv_device_uuid\":\""
        << JsonEscape(context.first_kv_device_uuid) << "\","
        << "\"first_kv_num_blocks\":" << context.first_kv_num_blocks << ","
        << "\"first_kv_block_size\":" << context.first_kv_block_size << ","
        << "\"first_kv_ipc_handle_bytes\":" << context.first_kv_ipc_handle_bytes
        << ","
        << "\"first_kv_storage_bytes\":" << context.first_kv_storage_bytes
        << ","
        << "\"first_kv_storage_offset_bytes\":"
        << context.first_kv_storage_offset_bytes << ","
        << "\"first_kv_event_handle_bytes\":"
        << context.first_kv_event_handle_bytes << ","
        << "\"first_kv_event_sync_required\":"
        << (context.first_kv_event_sync_required ? "true" : "false") << "}";
  }
  out << "]";
  return out.str();
}

std::string NativeServer::LastBlockAllocationJson() const {
  std::lock_guard<std::mutex> lock(block_allocations_mu_);
  if (!last_block_allocation_.valid) {
    return "null";
  }
  std::ostringstream out;
  out << "{"
      << "\"instance_id\":" << last_block_allocation_.instance_id << ","
      << "\"model_name\":\"" << JsonEscape(last_block_allocation_.model_name)
      << "\","
      << "\"record_count\":" << last_block_allocation_.record_count << ","
      << "\"last_request_id\":\""
      << JsonEscape(last_block_allocation_.last_request_id) << "\","
      << "\"last_new_block_count\":"
      << last_block_allocation_.last_new_block_count << ","
      << "\"last_new_token_count\":"
      << last_block_allocation_.last_new_token_count << "}";
  return out.str();
}

HttpResponse NativeServer::HandleLogLevelHttp(const std::string& path) {
  const auto params = QueryParams(path);
  const auto logger_it = params.find("logger_name");
  const auto level_it = params.find("level");
  const std::string logger_name =
      logger_it == params.end() ? "" : logger_it->second;
  const std::string level = level_it == params.end() ? "" : level_it->second;

  if (logger_name.empty() && level.empty()) {
    std::ostringstream out;
    out << "=== Loggers and Levels ===\n";
    std::lock_guard<std::mutex> lock(log_levels_mu_);
    if (log_levels_.empty()) {
      out << "lmcache.native: NOTSET\n";
    } else {
      for (const auto& [name, saved_level] : log_levels_) {
        out << name << ": " << saved_level << "\n";
      }
    }
    return {.status = 200, .content_type = "text/plain", .body = out.str()};
  }

  if (!logger_name.empty() && level.empty()) {
    std::lock_guard<std::mutex> lock(log_levels_mu_);
    const auto saved_level = log_levels_.find(logger_name);
    return {.status = 200,
            .content_type = "text/plain",
            .body = logger_name + ": " +
                    (saved_level == log_levels_.end() ? "NOTSET"
                                                      : saved_level->second)};
  }

  if (!logger_name.empty() && !level.empty()) {
    const std::string canonical_level = UppercaseAscii(level);
    if (!IsSupportedLogLevel(canonical_level)) {
      return {.status = 400,
              .content_type = "text/plain",
              .body = "Invalid log level: " + level};
    }
    {
      std::lock_guard<std::mutex> lock(log_levels_mu_);
      log_levels_[logger_name] = canonical_level;
    }
    return {.status = 200,
            .content_type = "text/plain",
            .body = "Set " + logger_name + " level to " + canonical_level +
                    " (including all handlers)"};
  }

  return {.status = 400,
          .content_type = "text/plain",
          .body = "logger_name is required when level is provided"};
}

HttpResponse NativeServer::HandleThreadsHttp(const std::string& path) {
  struct ThreadInfo {
    std::string name;
    std::string id;
  };

  const auto params = QueryParams(path);
  const auto name_it = params.find("name");
  const auto thread_id_it = params.find("thread_id");
  const std::string name_filter =
      name_it == params.end() ? "" : LowercaseAscii(name_it->second);
  const std::string thread_id_filter =
      thread_id_it == params.end() ? "" : thread_id_it->second;

  const auto thread_id_string = [](std::thread::id id) {
    std::ostringstream out;
    out << id;
    return out.str();
  };

  std::vector<ThreadInfo> threads;
  threads.push_back(
      {"native-http", thread_id_string(std::this_thread::get_id())});
  if (zmq_thread_.joinable()) {
    threads.push_back(
        {"native-zmq-loop", thread_id_string(zmq_thread_.get_id())});
  }
  for (std::size_t i = 0; i < workers_.size(); ++i) {
    if (!workers_[i].joinable()) {
      continue;
    }
    threads.push_back({"native-worker-" + std::to_string(i),
                       thread_id_string(workers_[i].get_id())});
  }

  std::vector<ThreadInfo> filtered_threads;
  for (const auto& thread : threads) {
    if (!name_filter.empty() &&
        LowercaseAscii(thread.name).find(name_filter) == std::string::npos) {
      continue;
    }
    if (!thread_id_filter.empty() && thread.id != thread_id_filter) {
      continue;
    }
    filtered_threads.push_back(thread);
  }

  std::ostringstream out;
  bool first = true;
  for (const auto& thread : filtered_threads) {
    if (!first) {
      out << "\n\n";
    }
    first = false;
    out << "Thread: " << thread.name << " (id=" << thread.id << ")\n"
        << "Stack trace unavailable in native MP server\n";
  }
  out << "\n\n=== Thread Summary ===\n"
      << "Total threads: " << filtered_threads.size() << "\n";

  return {.status = 200, .content_type = "text/plain", .body = out.str()};
}

HttpResponse NativeServer::HandlePeriodicThreadsHttp(const std::string& path) {
  const std::string route_path = PathWithoutQuery(path);
  if (route_path == "/periodic-threads-health") {
    return {.status = 200,
            .content_type = "application/json",
            .body =
                "{\"healthy\":true,\"unhealthy_count\":0,"
                "\"unhealthy_threads\":[]}"};
  }

  if (route_path == "/periodic-threads") {
    const auto params = QueryParams(path);
    const auto level_it = params.find("level");
    if (level_it != params.end() && !level_it->second.empty()) {
      const std::string level = LowercaseAscii(level_it->second);
      if (level != "critical" && level != "high" && level != "medium" &&
          level != "low") {
        return {.status = 400,
                .content_type = "application/json",
                .body = "{\"error\":\"Invalid level: " +
                        JsonEscape(level_it->second) +
                        ". Valid values: critical, high, medium, low\"}"};
      }
    }
    return {.status = 200,
            .content_type = "application/json",
            .body =
                "{\"summary\":{\"total_count\":0,\"running_count\":0,"
                "\"active_count\":0,\"by_level\":{\"critical\":{\"total\":0,"
                "\"running\":0,\"active\":0},\"high\":{\"total\":0,"
                "\"running\":0,\"active\":0},\"medium\":{\"total\":0,"
                "\"running\":0,\"active\":0},\"low\":{\"total\":0,"
                "\"running\":0,\"active\":0}}},\"threads\":[]}"};
  }

  const std::string prefix = "/periodic-threads/";
  const std::string thread_name =
      route_path.size() > prefix.size()
          ? UrlDecode(route_path.substr(prefix.size()))
          : "";
  return {.status = 404,
          .content_type = "application/json",
          .body = "{\"error\":\"Thread not found: " + JsonEscape(thread_name) +
                  "\"}"};
}

HttpResponse NativeServer::HandleQuotaHttp(const std::string& method,
                                           const std::string& path,
                                           const std::string& body) {
  const std::string route_path = PathWithoutQuery(path);
  std::unordered_map<std::string, std::uint64_t> usage;
  {
    std::lock_guard<std::mutex> lock(l2_adapters_mu_);
    for (const auto& adapter : l2_adapters_) {
      for (const auto& [salt, used_bytes] : adapter->UsageBytesByCacheSalt()) {
        usage[salt] += used_bytes;
      }
    }
  }

  if (route_path == "/quota") {
    if (method != "GET") {
      return {.status = 405,
              .content_type = "application/json",
              .body = "{\"error\":\"method not allowed\"}"};
    }
    std::ostringstream out;
    out << "{\"users\":{";
    bool first = true;
    {
      std::lock_guard<std::mutex> lock(quota_mu_);
      for (const auto& [salt, limit_bytes] : quota_limits_) {
        if (!first) {
          out << ",";
        }
        first = false;
        const std::uint64_t used_bytes =
            usage.count(salt) == 0 ? 0 : usage[salt];
        out << "\"" << JsonEscape(EscapeQuotaSalt(salt)) << "\":{"
            << "\"limit_gb\":" << JsonNumber(BytesToGb(limit_bytes)) << ","
            << "\"current_usage_gb\":" << JsonNumber(BytesToGb(used_bytes))
            << "}";
      }
    }
    out << "}}";
    return {
        .status = 200, .content_type = "application/json", .body = out.str()};
  }

  const std::string prefix = "/quota/";
  const std::string path_salt =
      route_path.size() > prefix.size()
          ? UrlDecode(route_path.substr(prefix.size()))
          : "";
  const std::string salt = UnescapeQuotaSalt(path_salt);
  if (path_salt.empty()) {
    return {.status = 404,
            .content_type = "application/json",
            .body = "{\"error\":\"not found\"}"};
  }

  if (method == "GET") {
    std::uint64_t limit_bytes = 0;
    bool exists = false;
    {
      std::lock_guard<std::mutex> lock(quota_mu_);
      const auto it = quota_limits_.find(salt);
      exists = it != quota_limits_.end();
      if (exists) {
        limit_bytes = it->second;
      }
    }
    const std::uint64_t used_bytes = usage.count(salt) == 0 ? 0 : usage[salt];
    std::ostringstream out;
    out << "{\"cache_salt\":\"" << JsonEscape(EscapeQuotaSalt(salt)) << "\","
        << "\"limit_gb\":" << JsonNumber(BytesToGb(limit_bytes)) << ","
        << "\"current_usage_gb\":" << JsonNumber(BytesToGb(used_bytes)) << ","
        << "\"exists\":" << (exists ? "true" : "false") << "}";
    return {
        .status = 200, .content_type = "application/json", .body = out.str()};
  }

  if (method == "DELETE") {
    bool removed = false;
    {
      std::lock_guard<std::mutex> lock(quota_mu_);
      removed = quota_limits_.erase(salt) > 0;
    }
    return {.status = 200,
            .content_type = "application/json",
            .body = "{\"cache_salt\":\"" + JsonEscape(EscapeQuotaSalt(salt)) +
                    "\",\"status\":\"" + (removed ? "removed" : "not_found") +
                    "\"}"};
  }

  if (method == "PUT") {
    std::string error;
    const auto limit_gb = JsonLimitGbField(body, &error);
    if (!limit_gb) {
      return {.status = 400,
              .content_type = "application/json",
              .body = "{\"error\":\"" + JsonEscape(error) + "\"}"};
    }
    const std::uint64_t limit_bytes =
        static_cast<std::uint64_t>(*limit_gb * 1024.0 * 1024.0 * 1024.0);
    {
      std::lock_guard<std::mutex> lock(quota_mu_);
      quota_limits_[salt] = limit_bytes;
    }
    return {.status = 200,
            .content_type = "application/json",
            .body = "{\"cache_salt\":\"" + JsonEscape(EscapeQuotaSalt(salt)) +
                    "\",\"limit_gb\":" + JsonNumber(*limit_gb) +
                    ",\"status\":\"ok\"}"};
  }

  return {.status = 405,
          .content_type = "application/json",
          .body = "{\"error\":\"method not allowed\"}"};
}

HttpResponse NativeServer::HandleKvCacheCheckHttp(const std::string& path) {
  const auto params = QueryParams(path);
  std::uint64_t instance_id = 0;
  if (const auto it = params.find("instance_id"); it != params.end()) {
    const auto parsed = ParseUnsignedText(it->second);
    if (!parsed) {
      return {.status = 400,
              .content_type = "application/json",
              .body = "{\"error\":\"instance_id must be non-negative\"}"};
    }
    instance_id = *parsed;
  }

  std::optional<RegisteredContext> context;
  {
    std::lock_guard<std::mutex> lock(registered_contexts_mu_);
    const auto it = registered_contexts_.find(instance_id);
    if (it != registered_contexts_.end()) {
      context = it->second;
    }
  }
  if (!context) {
    return {.status = 404,
            .content_type = "application/json",
            .body = "{\"error\":\"instance_id " + std::to_string(instance_id) +
                    " not registered\"}"};
  }

  const auto block_ids_it = params.find("block_ids");
  if (block_ids_it == params.end() || block_ids_it->second.empty()) {
    return {.status = 400,
            .content_type = "application/json",
            .body = "{\"error\":\"block_ids is required\"}"};
  }
  const auto parsed_blocks = ParseMixedBlockIds(block_ids_it->second);
  if (!parsed_blocks) {
    return {.status = 400,
            .content_type = "application/json",
            .body = "{\"error\":\"Invalid block_ids format\"}"};
  }

  const auto chunk_size_it = params.find("chunk_size");
  if (chunk_size_it == params.end()) {
    return {.status = 400,
            .content_type = "application/json",
            .body = "{\"error\":\"chunk_size must be positive\"}"};
  }
  const auto parsed_chunk_size = ParseUnsignedText(chunk_size_it->second);
  if (!parsed_chunk_size || *parsed_chunk_size == 0) {
    return {.status = 400,
            .content_type = "application/json",
            .body = "{\"error\":\"chunk_size must be positive\"}"};
  }
  if (*parsed_chunk_size > std::numeric_limits<std::uint32_t>::max()) {
    return {.status = 400,
            .content_type = "application/json",
            .body = "{\"error\":\"chunk_size is too large\"}"};
  }

  if (context->kv_wrappers.empty()) {
    return {.status = 404,
            .content_type = "application/json",
            .body = "{\"error\":\"kv_caches empty\"}"};
  }

  bool layerwise = false;
  if (const auto layerwise_it = params.find("layerwise");
      layerwise_it != params.end()) {
    const std::string normalized = LowercaseAscii(layerwise_it->second);
    layerwise = normalized == "1" || normalized == "true" ||
                normalized == "yes" || normalized == "on";
  }

  KvChecksumRequest request;
  request.gpu_block_ids = *parsed_blocks;
  request.chunk_blocks = static_cast<std::uint32_t>(*parsed_chunk_size);
  request.kv_layout = context->kv_layout;
  request.layerwise = layerwise;
  request.trt_llm_layout_hints = context->trt_llm_layout_hints;
  request.trt_llm_num_kv_heads = context->trt_llm_num_kv_heads;
  request.trt_llm_tokens_per_block = context->trt_llm_tokens_per_block;
  request.trt_llm_head_dim = context->trt_llm_head_dim;
  request.tensors.reserve(context->kv_wrappers.size());
  for (const msgpack::DecodedCudaIpcWrapper& wrapper : context->kv_wrappers) {
    request.tensors.push_back(KvTensorMetadata{
        .kind = wrapper.kind,
        .dtype = wrapper.dtype,
        .shape = wrapper.shape,
        .stride = wrapper.stride,
        .storage_offset = wrapper.storage_offset,
        .device_uuid = wrapper.device_uuid,
        .storage_bytes = wrapper.storage_bytes != 0 ? wrapper.storage_bytes
                                                    : wrapper.raw_nbytes,
        .storage_offset_bytes = wrapper.storage_offset_bytes,
        .ipc_handle = wrapper.ipc_handle,
        .event_handle = wrapper.event_handle,
        .event_sync_required = wrapper.event_sync_required,
    });
  }

  const KvChecksumResult result = ChecksumKvCacheBlocksFromCuda(request);
  if (!result.success) {
    const int status =
        StartsWith(result.error, "checksum not supported") ? 501 : 500;
    return {.status = status,
            .content_type = "application/json",
            .body = "{\"error\":\"" + JsonEscape(result.error) + "\"}"};
  }

  std::ostringstream out;
  out << "{\"status\":\"success\","
      << "\"chunk_size\":" << *parsed_chunk_size << ","
      << "\"num_chunks\":" << result.num_chunks << ","
      << "\"chunk_checksums\":";
  if (layerwise) {
    out << "{";
    for (std::size_t layer = 0; layer < result.layerwise_chunk_checksums.size();
         ++layer) {
      if (layer != 0) {
        out << ",";
      }
      out << "\"layer_" << layer << "\":[";
      const std::vector<std::string>& checksums =
          result.layerwise_chunk_checksums[layer];
      for (std::size_t i = 0; i < checksums.size(); ++i) {
        if (i != 0) {
          out << ",";
        }
        out << "\"" << JsonEscape(checksums[i]) << "\"";
      }
      out << "]";
    }
    out << "}";
  } else {
    out << "[";
    for (std::size_t i = 0; i < result.chunk_checksums.size(); ++i) {
      if (i != 0) {
        out << ",";
      }
      out << "\"" << JsonEscape(result.chunk_checksums[i]) << "\"";
    }
    out << "]";
  }
  out << ",\"layerwise\":" << (layerwise ? "true" : "false")
      << ",\"block_id_ranges\":\""
      << JsonEscape(CompressBlockIds(*parsed_blocks)) << "\"}";
  return {.status = 200, .content_type = "application/json", .body = out.str()};
}

HttpResponse NativeServer::HandleHttp(const std::string& method,
                                      const std::string& path,
                                      const std::string& body) {
  const std::string route_path = PathWithoutQuery(path);
  if (method == "GET" && route_path == "/") {
    return {.status = 200,
            .content_type = "application/json",
            .body = "{\"status\":\"ok\",\"service\":\"LMCache HTTP API\"}"};
  }
  if (method == "GET" && route_path == "/healthcheck") {
    return {.status = 200,
            .content_type = "application/json",
            .body = "{\"status\":\"healthy\"}"};
  }
  if (method == "GET" && route_path == "/status") {
    return {.status = 200,
            .content_type = "application/json",
            .body = StatusJson()};
  }
  if (method == "GET" && route_path == "/conf") {
    return {.status = 200,
            .content_type = "application/json",
            .body = ConfigJson()};
  }
  if (method == "GET" && route_path == "/lmc_version") {
    return {.status = 200,
            .content_type = "application/json",
            .body = JsonStringValue(config_.lmcache_version)};
  }
  if (method == "GET" && route_path == "/commit_id") {
    return {.status = 200,
            .content_type = "application/json",
            .body = JsonStringValue(config_.lmcache_commit_id)};
  }
  if (method == "GET" && route_path == "/version") {
    const std::string version =
        config_.lmcache_version.empty() ? "NA" : config_.lmcache_version;
    const std::string commit_id =
        config_.lmcache_commit_id.empty() ? "NA" : config_.lmcache_commit_id;
    return {.status = 200,
            .content_type = "application/json",
            .body = JsonStringValue(version + "-" + commit_id)};
  }
  if (method == "GET" && route_path == "/metrics") {
    return {.status = 200,
            .content_type = "text/plain; version=0.0.4; charset=utf-8",
            .body = PrometheusMetricsText()};
  }
  if (method == "GET" && route_path == "/env") {
    return {.status = 200,
            .content_type = "text/plain",
            .body = EnvironmentJsonText()};
  }
  if (method == "POST" && route_path == "/clear-cache") {
    ForceClearCache();
    return {.status = 200,
            .content_type = "application/json",
            .body = "{\"status\":\"ok\"}"};
  }
  if (method == "POST" && route_path == "/metrics/reset") {
    ResetMetrics();
    return {.status = 200, .content_type = "text/plain", .body = "ok"};
  }
  if (method == "GET" && route_path == "/loglevel") {
    return HandleLogLevelHttp(path);
  }
  if (method == "GET" && route_path == "/threads") {
    return HandleThreadsHttp(path);
  }
  if (method == "GET" && (route_path == "/periodic-threads" ||
                          StartsWith(route_path, "/periodic-threads/") ||
                          route_path == "/periodic-threads-health")) {
    return HandlePeriodicThreadsHttp(path);
  }
  if (route_path == "/quota" || StartsWith(route_path, "/quota/")) {
    return HandleQuotaHttp(method, path, body);
  }
  if (method == "GET" && route_path == "/kvcache/check") {
    return HandleKvCacheCheckHttp(path);
  }
  if (route_path == "/" || route_path == "/healthcheck" ||
      route_path == "/status" || route_path == "/clear-cache" ||
      route_path == "/conf" || route_path == "/lmc_version" ||
      route_path == "/commit_id" || route_path == "/version" ||
      route_path == "/metrics" || route_path == "/metrics/reset" ||
      route_path == "/env" || route_path == "/loglevel" ||
      route_path == "/threads" || route_path == "/periodic-threads" ||
      StartsWith(route_path, "/periodic-threads/") ||
      route_path == "/periodic-threads-health" ||
      route_path == "/kvcache/check") {
    return {.status = 405,
            .content_type = "application/json",
            .body = "{\"error\":\"method not allowed\"}"};
  }
  return {.status = 404,
          .content_type = "application/json",
          .body = "{\"error\":\"not found\"}"};
}

std::string NativeServer::StatusJson() const {
  const LmcacheMpCppStats stats = lmcache_mp_cpp_cache_stats(cache_);
  std::size_t worker_queue_depth = 0;
  {
    std::lock_guard<std::mutex> lock(tasks_mu_);
    worker_queue_depth = QueuedTaskCountLocked();
  }
  std::size_t response_queue_depth = 0;
  {
    std::lock_guard<std::mutex> lock(responses_mu_);
    response_queue_depth = responses_.size();
  }
  const std::size_t observed_client_count = ObservedClientCount();
  const std::uint64_t cache_hits = AtomicLoad(metrics_.cache_hits);
  const std::uint64_t cache_misses = AtomicLoad(metrics_.cache_misses);
  const std::uint64_t cache_lookups = cache_hits + cache_misses;
  const CudaTransferDeviceCacheStats device_cache_stats =
      GetCudaTransferDeviceCacheStats();
  const double cache_hit_rate = cache_lookups == 0
                                    ? 0.0
                                    : static_cast<double>(cache_hits) /
                                          static_cast<double>(cache_lookups);
  std::ostringstream out;
  out << "{"
      << "\"native\":true,"
      << "\"protocol_version\":" << LMCACHE_MP_PROTOCOL_VERSION << ","
      << "\"cuda_transfer_enabled\":"
      << (NativeCudaTransferEnabled() ? "true" : "false") << ","
      << "\"cuda_gpu_hot_cache_enabled\":"
      << (config_.enable_cuda_gpu_hot_cache ? "true" : "false") << ","
      << "\"cuda_gpu_hot_cache\":{"
      << "\"entries\":" << device_cache_stats.entries << ","
      << "\"bytes\":" << device_cache_stats.bytes << "},"
      << "\"chunk_size\":" << config_.chunk_size << ","
      << "\"registered_context_count\":" << RegisteredContextCount() << ","
      << "\"registered_contexts\":" << RegisteredContextsJson() << ","
      << "\"last_block_allocation\":" << LastBlockAllocationJson() << ","
      << "\"num_l2_adapters\":" << l2_adapters_.size() << ","
      << "\"eviction_policy\":\"" << JsonEscape(config_.eviction_policy)
      << "\","
      << "\"cache\":{"
      << "\"dram_bytes\":" << stats.dram_bytes << ","
      << "\"dram_capacity_bytes\":" << config_.dram_capacity_bytes << ","
      << "\"disk_bytes\":" << stats.disk_bytes << ","
      << "\"dram_entries\":" << stats.dram_entries << ","
      << "\"disk_entries\":" << stats.disk_entries << ","
      << "\"total_entries\":" << stats.total_entries << ","
      << "\"locked_entries\":" << stats.locked_entries << ","
      << "\"lock_count\":" << stats.lock_count << ","
      << "\"locked_bytes\":" << stats.locked_bytes << ","
      << "\"pinned_entries\":" << stats.pinned_entries << ","
      << "\"eviction_count\":" << stats.eviction_count << "},"
      << "\"l2_adapters\":[";
  for (std::size_t i = 0; i < l2_adapters_.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    const L2AdapterStatus adapter_status = l2_adapters_[i]->Status();
    out << "{"
        << "\"type\":\"" << JsonEscape(adapter_status.type) << "\","
        << "\"base_path\":\"" << JsonEscape(adapter_status.base_path) << "\","
        << "\"stored_files\":" << adapter_status.stored_files << ","
        << "\"stored_bytes\":" << adapter_status.stored_bytes << "}";
  }
  out << "],"
      << "\"metrics\":{"
      << "\"request_count\":" << AtomicLoad(metrics_.request_count) << ","
      << "\"unsupported_count\":" << AtomicLoad(metrics_.unsupported_count)
      << ","
      << "\"clear_count\":" << AtomicLoad(metrics_.clear_count) << ","
      << "\"store_count\":" << AtomicLoad(metrics_.store_count) << ","
      << "\"retrieve_count\":" << AtomicLoad(metrics_.retrieve_count) << ","
      << "\"lookup_count\":" << AtomicLoad(metrics_.lookup_count) << ","
      << "\"lookup_result_fast_path_count\":"
      << AtomicLoad(metrics_.lookup_result_fast_path_count) << ","
      << "\"invalid_payload_count\":"
      << AtomicLoad(metrics_.invalid_payload_count) << ","
      << "\"block_allocation_report_count\":"
      << AtomicLoad(metrics_.block_allocation_report_count) << ","
      << "\"block_allocation_record_count\":"
      << AtomicLoad(metrics_.block_allocation_record_count) << ","
      << "\"cache_hits\":" << cache_hits << ","
      << "\"cache_misses\":" << cache_misses << ","
      << "\"cache_hit_rate\":" << cache_hit_rate << ","
      << "\"partial_hit_count\":" << AtomicLoad(metrics_.partial_hit_count)
      << ","
      << "\"l1_hit_count\":" << AtomicLoad(metrics_.l1_hit_count) << ","
      << "\"l2_hit_count\":" << AtomicLoad(metrics_.l2_hit_count) << ","
      << "\"l2_miss_count\":" << AtomicLoad(metrics_.l2_miss_count) << ","
      << "\"l2_store_count\":" << AtomicLoad(metrics_.l2_store_count) << ","
      << "\"l2_load_count\":" << AtomicLoad(metrics_.l2_load_count) << ","
      << "\"l2_error_count\":" << AtomicLoad(metrics_.l2_error_count) << ","
      << "\"transfer_lock_count\":" << AtomicLoad(metrics_.transfer_lock_count)
      << ","
      << "\"transfer_lock_failure_count\":"
      << AtomicLoad(metrics_.transfer_lock_failure_count) << ","
      << "\"transfer_lock_wait_total_us\":"
      << AtomicLoad(metrics_.transfer_lock_wait_total_us) << ","
      << "\"transfer_lock_wait_max_us\":"
      << AtomicLoad(metrics_.transfer_lock_wait_max_us) << ","
      << "\"transfer_lock_hold_total_us\":"
      << AtomicLoad(metrics_.transfer_lock_hold_total_us) << ","
      << "\"transfer_lock_hold_max_us\":"
      << AtomicLoad(metrics_.transfer_lock_hold_max_us) << ","
      << "\"worker_count\":" << workers_.size() << ","
      << "\"active_client_count\":" << observed_client_count << ","
      << "\"observed_client_count\":" << observed_client_count << ","
      << "\"active_worker_count\":" << AtomicLoad(metrics_.active_worker_count)
      << ","
      << "\"worker_queue_depth\":" << worker_queue_depth << ","
      << "\"max_worker_queue_depth\":" << config_.max_queued_tasks << ","
      << "\"response_queue_depth\":" << response_queue_depth << ","
      << "\"queue_full_count\":" << AtomicLoad(metrics_.queue_full_count) << ","
      << "\"request_queue_wait_count\":"
      << AtomicLoad(metrics_.request_queue_wait_count) << ","
      << "\"request_queue_wait_total_us\":"
      << AtomicLoad(metrics_.request_queue_wait_total_us) << ","
      << "\"request_queue_wait_max_us\":"
      << AtomicLoad(metrics_.request_queue_wait_max_us) << ","
      << "\"request_type_queue_wait\":{"
      << "\"lookup\":{\"count\":"
      << AtomicLoad(metrics_.lookup_request_queue_wait_count)
      << ",\"total_us\":"
      << AtomicLoad(metrics_.lookup_request_queue_wait_total_us)
      << ",\"max_us\":"
      << AtomicLoad(metrics_.lookup_request_queue_wait_max_us) << "},"
      << "\"store\":{\"count\":"
      << AtomicLoad(metrics_.store_request_queue_wait_count)
      << ",\"total_us\":"
      << AtomicLoad(metrics_.store_request_queue_wait_total_us)
      << ",\"max_us\":"
      << AtomicLoad(metrics_.store_request_queue_wait_max_us) << "},"
      << "\"retrieve\":{\"count\":"
      << AtomicLoad(metrics_.retrieve_request_queue_wait_count)
      << ",\"total_us\":"
      << AtomicLoad(metrics_.retrieve_request_queue_wait_total_us)
      << ",\"max_us\":"
      << AtomicLoad(metrics_.retrieve_request_queue_wait_max_us) << "},"
      << "\"free_lookup_locks\":{\"count\":"
      << AtomicLoad(metrics_.free_lookup_locks_request_queue_wait_count)
      << ",\"total_us\":"
      << AtomicLoad(metrics_.free_lookup_locks_request_queue_wait_total_us)
      << ",\"max_us\":"
      << AtomicLoad(metrics_.free_lookup_locks_request_queue_wait_max_us)
      << "}},"
      << "\"request_latency_count\":"
      << AtomicLoad(metrics_.request_latency_count) << ","
      << "\"request_latency_total_us\":"
      << AtomicLoad(metrics_.request_latency_total_us) << ","
      << "\"request_latency_max_us\":"
      << AtomicLoad(metrics_.request_latency_max_us) << ","
      << "\"request_type_latency\":{"
      << "\"lookup\":{\"count\":"
      << AtomicLoad(metrics_.lookup_request_latency_count)
      << ",\"total_us\":"
      << AtomicLoad(metrics_.lookup_request_latency_total_us) << ",\"max_us\":"
      << AtomicLoad(metrics_.lookup_request_latency_max_us) << "},"
      << "\"store\":{\"count\":"
      << AtomicLoad(metrics_.store_request_latency_count)
      << ",\"total_us\":"
      << AtomicLoad(metrics_.store_request_latency_total_us) << ",\"max_us\":"
      << AtomicLoad(metrics_.store_request_latency_max_us) << "},"
      << "\"retrieve\":{\"count\":"
      << AtomicLoad(metrics_.retrieve_request_latency_count)
      << ",\"total_us\":"
      << AtomicLoad(metrics_.retrieve_request_latency_total_us)
      << ",\"max_us\":"
      << AtomicLoad(metrics_.retrieve_request_latency_max_us) << "},"
      << "\"free_lookup_locks\":{\"count\":"
      << AtomicLoad(metrics_.free_lookup_locks_request_latency_count)
      << ",\"total_us\":"
      << AtomicLoad(metrics_.free_lookup_locks_request_latency_total_us)
      << ",\"max_us\":"
      << AtomicLoad(metrics_.free_lookup_locks_request_latency_max_us) << "}},"
      << "\"cuda_transfer_store_total_us\":"
      << AtomicLoad(metrics_.cuda_transfer_store_total_us) << ","
      << "\"cuda_transfer_store_max_us\":"
      << AtomicLoad(metrics_.cuda_transfer_store_max_us) << ","
      << "\"cuda_transfer_retrieve_total_us\":"
      << AtomicLoad(metrics_.cuda_transfer_retrieve_total_us) << ","
      << "\"cuda_transfer_retrieve_max_us\":"
      << AtomicLoad(metrics_.cuda_transfer_retrieve_max_us) << ","
      << "\"cuda_transfer_wait_event_total_us\":"
      << AtomicLoad(metrics_.cuda_transfer_wait_event_total_us) << ","
      << "\"cuda_transfer_open_tensors_total_us\":"
      << AtomicLoad(metrics_.cuda_transfer_open_tensors_total_us) << ","
      << "\"cuda_transfer_copy_total_us\":"
      << AtomicLoad(metrics_.cuda_transfer_copy_total_us) << ","
      << "\"cuda_transfer_cache_total_us\":"
      << AtomicLoad(metrics_.cuda_transfer_cache_total_us) << ","
      << "\"cuda_transfer_completion_event_total_us\":"
      << AtomicLoad(metrics_.cuda_transfer_completion_event_total_us) << ","
      << "\"cuda_transfer_bytes\":"
      << AtomicLoad(metrics_.cuda_transfer_bytes) << ","
      << "\"cuda_transfer_memcpy_calls\":"
      << AtomicLoad(metrics_.cuda_transfer_memcpy_calls) << ","
      << "\"cuda_transfer_kernel_calls\":"
      << AtomicLoad(metrics_.cuda_transfer_kernel_calls) << ","
      << "\"request_latency_histogram\":{"
      << "\"le_100us\":" << AtomicLoad(metrics_.request_latency_le_100us) << ","
      << "\"le_500us\":" << AtomicLoad(metrics_.request_latency_le_500us) << ","
      << "\"le_1ms\":" << AtomicLoad(metrics_.request_latency_le_1ms) << ","
      << "\"le_5ms\":" << AtomicLoad(metrics_.request_latency_le_5ms) << ","
      << "\"le_10ms\":" << AtomicLoad(metrics_.request_latency_le_10ms) << ","
      << "\"le_50ms\":" << AtomicLoad(metrics_.request_latency_le_50ms) << ","
      << "\"le_100ms\":" << AtomicLoad(metrics_.request_latency_le_100ms) << ","
      << "\"gt_100ms\":" << AtomicLoad(metrics_.request_latency_gt_100ms)
      << "}}"
      << "}";
  return out.str();
}

std::string NativeServer::ConfigJson() const {
  std::ostringstream out;
  out << "{"
      << "\"native\":true,"
      << "\"version\":{"
      << "\"lmcache_version\":\"" << JsonEscape(config_.lmcache_version)
      << "\","
      << "\"commit_id\":\"" << JsonEscape(config_.lmcache_commit_id) << "\"},"
      << "\"mp\":{"
      << "\"host\":\"" << JsonEscape(config_.host) << "\","
      << "\"port\":" << config_.port << ","
      << "\"chunk_size\":" << config_.chunk_size << ","
      << "\"max_workers\":" << config_.max_workers << ","
      << "\"max_queued_tasks\":" << config_.max_queued_tasks << ","
      << "\"log_level\":\"" << JsonEscape(config_.startup_log_level) << "\","
      << "\"eviction_policy\":\"" << JsonEscape(config_.eviction_policy)
      << "\","
      << "\"cuda_gpu_hot_cache_enabled\":"
      << (config_.enable_cuda_gpu_hot_cache ? "true" : "false")
      << "},"
      << "\"http\":{"
      << "\"http_host\":\"" << JsonEscape(config_.http_host) << "\","
      << "\"http_port\":" << config_.http_port << ","
      << "\"enabled\":" << (config_.enable_http ? "true" : "false") << "},"
      << "\"storage_manager\":{"
      << "\"dram_capacity_bytes\":" << config_.dram_capacity_bytes << ","
      << "\"disk_path\":\"" << JsonEscape(config_.disk_path) << "\","
      << "\"l2_adapters\":[";
  for (std::size_t i = 0; i < l2_adapters_.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    const L2AdapterStatus adapter_status = l2_adapters_[i]->Status();
    out << "{"
        << "\"type\":\"" << JsonEscape(adapter_status.type) << "\","
        << "\"base_path\":\"" << JsonEscape(adapter_status.base_path) << "\"}";
  }
  out << "]},"
      << "\"observability\":{"
      << "\"status_endpoint\":\"/status\","
      << "\"metrics_endpoint\":\"/metrics\","
      << "\"metrics_reset_endpoint\":\"/metrics/reset\"}"
      << "}";
  return out.str();
}

std::string NativeServer::PrometheusMetricsText() const {
  const LmcacheMpCppStats stats = lmcache_mp_cpp_cache_stats(cache_);
  std::size_t worker_queue_depth = 0;
  {
    std::lock_guard<std::mutex> lock(tasks_mu_);
    worker_queue_depth = QueuedTaskCountLocked();
  }
  std::size_t response_queue_depth = 0;
  {
    std::lock_guard<std::mutex> lock(responses_mu_);
    response_queue_depth = responses_.size();
  }
  const std::size_t observed_client_count = ObservedClientCount();
  const std::uint64_t cache_hits = AtomicLoad(metrics_.cache_hits);
  const std::uint64_t cache_misses = AtomicLoad(metrics_.cache_misses);
  const std::uint64_t cache_lookups = cache_hits + cache_misses;
  const double cache_hit_rate = cache_lookups == 0
                                    ? 0.0
                                    : static_cast<double>(cache_hits) /
                                          static_cast<double>(cache_lookups);

  std::ostringstream out;
  AppendPrometheusMetric(&out, "lmcache_mp_native_requests_total", "counter",
                         "Total native MP ZMQ requests handled.",
                         AtomicLoad(metrics_.request_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_unsupported_requests_total",
                         "counter",
                         "Total native MP requests rejected or unsupported.",
                         AtomicLoad(metrics_.unsupported_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_invalid_payloads_total",
                         "counter",
                         "Total native MP requests with invalid payloads.",
                         AtomicLoad(metrics_.invalid_payload_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_stores_total", "counter",
                         "Total native MP STORE requests.",
                         AtomicLoad(metrics_.store_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_retrieves_total", "counter",
                         "Total native MP RETRIEVE requests.",
                         AtomicLoad(metrics_.retrieve_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_lookups_total", "counter",
                         "Total native MP LOOKUP requests.",
                         AtomicLoad(metrics_.lookup_count));
  AppendPrometheusMetric(&out,
                         "lmcache_mp_native_lookup_result_fast_path_total",
                         "counter",
                         "Total native MP lookup-result status responses "
                         "served without the worker queue.",
                         AtomicLoad(metrics_.lookup_result_fast_path_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_clears_total", "counter",
                         "Total native cache clear operations.",
                         AtomicLoad(metrics_.clear_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_cache_hits_total", "counter",
                         "Total native lookup object hits.", cache_hits);
  AppendPrometheusMetric(&out, "lmcache_mp_native_cache_misses_total",
                         "counter", "Total native lookup object misses.",
                         cache_misses);
  AppendPrometheusMetric(&out, "lmcache_mp_native_cache_hit_rate", "gauge",
                         "Native lookup object hit rate.", cache_hit_rate);
  AppendPrometheusMetric(&out, "lmcache_mp_native_partial_hits_total",
                         "counter", "Total native partial lookup hits.",
                         AtomicLoad(metrics_.partial_hit_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_l1_hits_total", "counter",
                         "Total native lookup object hits from L1.",
                         AtomicLoad(metrics_.l1_hit_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_l2_hits_total", "counter",
                         "Total native lookup object hits from L2 metadata.",
                         AtomicLoad(metrics_.l2_hit_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_l2_misses_total", "counter",
                         "Total native lookup object misses after L2 checks.",
                         AtomicLoad(metrics_.l2_miss_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_l2_stores_total", "counter",
                         "Total native L2 chunk store attempts.",
                         AtomicLoad(metrics_.l2_store_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_l2_loads_total", "counter",
                         "Total native L2 chunk load attempts.",
                         AtomicLoad(metrics_.l2_load_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_l2_errors_total", "counter",
                         "Total native L2 operation errors.",
                         AtomicLoad(metrics_.l2_error_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_transfer_locks_total",
                         "counter", "Total native transfer locks acquired.",
                         AtomicLoad(metrics_.transfer_lock_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_transfer_lock_failures_total",
                         "counter",
                         "Total native transfer lock acquisition failures.",
                         AtomicLoad(metrics_.transfer_lock_failure_count));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_transfer_lock_wait_total_microseconds",
      "counter", "Total native transfer lock acquisition time in microseconds.",
      AtomicLoad(metrics_.transfer_lock_wait_total_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_transfer_lock_wait_max_microseconds", "gauge",
      "Maximum native transfer lock acquisition time in microseconds.",
      AtomicLoad(metrics_.transfer_lock_wait_max_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_transfer_lock_hold_total_microseconds",
      "counter", "Total native transfer lock hold time in microseconds.",
      AtomicLoad(metrics_.transfer_lock_hold_total_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_transfer_lock_hold_max_microseconds", "gauge",
      "Maximum native transfer lock hold time in microseconds.",
      AtomicLoad(metrics_.transfer_lock_hold_max_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_block_allocation_reports_total", "counter",
      "Total native block allocation reports decoded.",
      AtomicLoad(metrics_.block_allocation_report_count));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_block_allocation_records_total", "counter",
      "Total native block allocation records decoded.",
      AtomicLoad(metrics_.block_allocation_record_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_request_queue_wait_count",
                         "counter",
                         "Total native MP request queue-wait observations.",
                         AtomicLoad(metrics_.request_queue_wait_count));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_request_queue_wait_total_microseconds",
      "counter", "Total native MP request queue wait in microseconds.",
      AtomicLoad(metrics_.request_queue_wait_total_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_request_queue_wait_max_microseconds", "gauge",
      "Maximum observed native MP request queue wait in microseconds.",
      AtomicLoad(metrics_.request_queue_wait_max_us));
  AppendRequestTypeQueueWaitMetrics(
      &out, "lookup", "LOOKUP",
      AtomicLoad(metrics_.lookup_request_queue_wait_count),
      AtomicLoad(metrics_.lookup_request_queue_wait_total_us),
      AtomicLoad(metrics_.lookup_request_queue_wait_max_us));
  AppendRequestTypeQueueWaitMetrics(
      &out, "store", "STORE",
      AtomicLoad(metrics_.store_request_queue_wait_count),
      AtomicLoad(metrics_.store_request_queue_wait_total_us),
      AtomicLoad(metrics_.store_request_queue_wait_max_us));
  AppendRequestTypeQueueWaitMetrics(
      &out, "retrieve", "RETRIEVE",
      AtomicLoad(metrics_.retrieve_request_queue_wait_count),
      AtomicLoad(metrics_.retrieve_request_queue_wait_total_us),
      AtomicLoad(metrics_.retrieve_request_queue_wait_max_us));
  AppendRequestTypeQueueWaitMetrics(
      &out, "free_lookup_locks", "FREE_LOOKUP_LOCKS",
      AtomicLoad(metrics_.free_lookup_locks_request_queue_wait_count),
      AtomicLoad(metrics_.free_lookup_locks_request_queue_wait_total_us),
      AtomicLoad(metrics_.free_lookup_locks_request_queue_wait_max_us));
  AppendPrometheusMetric(&out, "lmcache_mp_native_request_latency_count",
                         "counter",
                         "Total native MP request latency observations.",
                         AtomicLoad(metrics_.request_latency_count));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_request_latency_total_microseconds", "counter",
      "Total native MP request latency in microseconds.",
      AtomicLoad(metrics_.request_latency_total_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_request_latency_max_microseconds", "gauge",
      "Maximum observed native MP request latency in microseconds.",
      AtomicLoad(metrics_.request_latency_max_us));
  AppendRequestTypeLatencyMetrics(
      &out, "lookup", "LOOKUP",
      AtomicLoad(metrics_.lookup_request_latency_count),
      AtomicLoad(metrics_.lookup_request_latency_total_us),
      AtomicLoad(metrics_.lookup_request_latency_max_us));
  AppendRequestTypeLatencyMetrics(
      &out, "store", "STORE",
      AtomicLoad(metrics_.store_request_latency_count),
      AtomicLoad(metrics_.store_request_latency_total_us),
      AtomicLoad(metrics_.store_request_latency_max_us));
  AppendRequestTypeLatencyMetrics(
      &out, "retrieve", "RETRIEVE",
      AtomicLoad(metrics_.retrieve_request_latency_count),
      AtomicLoad(metrics_.retrieve_request_latency_total_us),
      AtomicLoad(metrics_.retrieve_request_latency_max_us));
  AppendRequestTypeLatencyMetrics(
      &out, "free_lookup_locks", "FREE_LOOKUP_LOCKS",
      AtomicLoad(metrics_.free_lookup_locks_request_latency_count),
      AtomicLoad(metrics_.free_lookup_locks_request_latency_total_us),
      AtomicLoad(metrics_.free_lookup_locks_request_latency_max_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_store_total_microseconds",
      "counter", "Total native CUDA STORE transfer time in microseconds.",
      AtomicLoad(metrics_.cuda_transfer_store_total_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_store_max_microseconds", "gauge",
      "Maximum native CUDA STORE transfer time in microseconds.",
      AtomicLoad(metrics_.cuda_transfer_store_max_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_retrieve_total_microseconds",
      "counter", "Total native CUDA RETRIEVE transfer time in microseconds.",
      AtomicLoad(metrics_.cuda_transfer_retrieve_total_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_retrieve_max_microseconds",
      "gauge", "Maximum native CUDA RETRIEVE transfer time in microseconds.",
      AtomicLoad(metrics_.cuda_transfer_retrieve_max_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_copy_total_microseconds",
      "counter", "Total native CUDA copy time in microseconds.",
      AtomicLoad(metrics_.cuda_transfer_copy_total_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_cache_total_microseconds",
      "counter", "Total native CUDA transfer cache time in microseconds.",
      AtomicLoad(metrics_.cuda_transfer_cache_total_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_open_tensors_total_microseconds",
      "counter",
      "Total native CUDA IPC tensor open time in microseconds.",
      AtomicLoad(metrics_.cuda_transfer_open_tensors_total_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_wait_event_total_microseconds",
      "counter", "Total native CUDA input event wait time in microseconds.",
      AtomicLoad(metrics_.cuda_transfer_wait_event_total_us));
  AppendPrometheusMetric(
      &out,
      "lmcache_mp_native_cuda_transfer_completion_event_total_microseconds",
      "counter",
      "Total native CUDA completion event creation time in microseconds.",
      AtomicLoad(metrics_.cuda_transfer_completion_event_total_us));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_bytes_total", "counter",
      "Total bytes copied by native CUDA transfer.",
      AtomicLoad(metrics_.cuda_transfer_bytes));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_memcpy_calls_total", "counter",
      "Total cudaMemcpy calls issued by native CUDA transfer.",
      AtomicLoad(metrics_.cuda_transfer_memcpy_calls));
  AppendPrometheusMetric(
      &out, "lmcache_mp_native_cuda_transfer_kernel_calls_total", "counter",
      "Total native CUDA block-transfer kernel calls.",
      AtomicLoad(metrics_.cuda_transfer_kernel_calls));

  AppendPrometheusMetric(&out, "lmcache_mp_native_cache_dram_bytes", "gauge",
                         "Native cache DRAM bytes in use.", stats.dram_bytes);
  AppendPrometheusMetric(&out, "lmcache_mp_native_cache_dram_capacity_bytes",
                         "gauge", "Native cache DRAM byte capacity.",
                         config_.dram_capacity_bytes);
  AppendPrometheusMetric(&out, "lmcache_mp_native_cache_disk_bytes", "gauge",
                         "Native cache disk-tier bytes in use.",
                         stats.disk_bytes);
  AppendPrometheusMetric(&out, "lmcache_mp_native_cache_locked_bytes", "gauge",
                         "Native cache bytes protected by locks.",
                         stats.locked_bytes);
  AppendPrometheusMetric(&out, "lmcache_mp_native_cache_locked_entries",
                         "gauge", "Native cache entries protected by locks.",
                         stats.locked_entries);
  AppendPrometheusMetric(&out, "lmcache_mp_native_cache_evictions_total",
                         "counter", "Total native cache LRU spill evictions.",
                         stats.eviction_count);
  AppendPrometheusMetric(&out, "lmcache_mp_native_workers", "gauge",
                         "Configured native worker count.", workers_.size());
  AppendPrometheusMetric(&out, "lmcache_mp_native_active_workers", "gauge",
                         "Native workers currently handling requests.",
                         AtomicLoad(metrics_.active_worker_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_worker_queue_depth", "gauge",
                         "Native worker request queue depth.",
                         worker_queue_depth);
  AppendPrometheusMetric(&out, "lmcache_mp_native_response_queue_depth",
                         "gauge", "Native response queue depth.",
                         response_queue_depth);
  AppendPrometheusMetric(&out, "lmcache_mp_native_queue_full_total", "counter",
                         "Native worker queue full events.",
                         AtomicLoad(metrics_.queue_full_count));
  AppendPrometheusMetric(&out, "lmcache_mp_native_observed_clients", "gauge",
                         "Valid ZMQ client identities observed since startup.",
                         observed_client_count);

  return out.str();
}

void NativeServer::ResetMetrics() {
  AtomicStoreZero(&metrics_.request_count);
  AtomicStoreZero(&metrics_.unsupported_count);
  AtomicStoreZero(&metrics_.clear_count);
  AtomicStoreZero(&metrics_.store_count);
  AtomicStoreZero(&metrics_.retrieve_count);
  AtomicStoreZero(&metrics_.lookup_count);
  AtomicStoreZero(&metrics_.lookup_result_fast_path_count);
  AtomicStoreZero(&metrics_.invalid_payload_count);
  AtomicStoreZero(&metrics_.block_allocation_report_count);
  AtomicStoreZero(&metrics_.block_allocation_record_count);
  AtomicStoreZero(&metrics_.cache_hits);
  AtomicStoreZero(&metrics_.cache_misses);
  AtomicStoreZero(&metrics_.partial_hit_count);
  AtomicStoreZero(&metrics_.l1_hit_count);
  AtomicStoreZero(&metrics_.l2_hit_count);
  AtomicStoreZero(&metrics_.l2_miss_count);
  AtomicStoreZero(&metrics_.l2_store_count);
  AtomicStoreZero(&metrics_.l2_load_count);
  AtomicStoreZero(&metrics_.l2_error_count);
  AtomicStoreZero(&metrics_.transfer_lock_count);
  AtomicStoreZero(&metrics_.transfer_lock_failure_count);
  AtomicStoreZero(&metrics_.transfer_lock_wait_total_us);
  AtomicStoreZero(&metrics_.transfer_lock_wait_max_us);
  AtomicStoreZero(&metrics_.transfer_lock_hold_total_us);
  AtomicStoreZero(&metrics_.transfer_lock_hold_max_us);
  AtomicStoreZero(&metrics_.queue_full_count);
  AtomicStoreZero(&metrics_.request_queue_wait_count);
  AtomicStoreZero(&metrics_.request_queue_wait_total_us);
  AtomicStoreZero(&metrics_.request_queue_wait_max_us);
  AtomicStoreZero(&metrics_.lookup_request_queue_wait_count);
  AtomicStoreZero(&metrics_.lookup_request_queue_wait_total_us);
  AtomicStoreZero(&metrics_.lookup_request_queue_wait_max_us);
  AtomicStoreZero(&metrics_.store_request_queue_wait_count);
  AtomicStoreZero(&metrics_.store_request_queue_wait_total_us);
  AtomicStoreZero(&metrics_.store_request_queue_wait_max_us);
  AtomicStoreZero(&metrics_.retrieve_request_queue_wait_count);
  AtomicStoreZero(&metrics_.retrieve_request_queue_wait_total_us);
  AtomicStoreZero(&metrics_.retrieve_request_queue_wait_max_us);
  AtomicStoreZero(&metrics_.free_lookup_locks_request_queue_wait_count);
  AtomicStoreZero(&metrics_.free_lookup_locks_request_queue_wait_total_us);
  AtomicStoreZero(&metrics_.free_lookup_locks_request_queue_wait_max_us);
  AtomicStoreZero(&metrics_.request_latency_count);
  AtomicStoreZero(&metrics_.request_latency_total_us);
  AtomicStoreZero(&metrics_.request_latency_max_us);
  AtomicStoreZero(&metrics_.lookup_request_latency_count);
  AtomicStoreZero(&metrics_.lookup_request_latency_total_us);
  AtomicStoreZero(&metrics_.lookup_request_latency_max_us);
  AtomicStoreZero(&metrics_.store_request_latency_count);
  AtomicStoreZero(&metrics_.store_request_latency_total_us);
  AtomicStoreZero(&metrics_.store_request_latency_max_us);
  AtomicStoreZero(&metrics_.retrieve_request_latency_count);
  AtomicStoreZero(&metrics_.retrieve_request_latency_total_us);
  AtomicStoreZero(&metrics_.retrieve_request_latency_max_us);
  AtomicStoreZero(&metrics_.free_lookup_locks_request_latency_count);
  AtomicStoreZero(&metrics_.free_lookup_locks_request_latency_total_us);
  AtomicStoreZero(&metrics_.free_lookup_locks_request_latency_max_us);
  AtomicStoreZero(&metrics_.request_latency_le_100us);
  AtomicStoreZero(&metrics_.request_latency_le_500us);
  AtomicStoreZero(&metrics_.request_latency_le_1ms);
  AtomicStoreZero(&metrics_.request_latency_le_5ms);
  AtomicStoreZero(&metrics_.request_latency_le_10ms);
  AtomicStoreZero(&metrics_.request_latency_le_50ms);
  AtomicStoreZero(&metrics_.request_latency_le_100ms);
  AtomicStoreZero(&metrics_.request_latency_gt_100ms);
  AtomicStoreZero(&metrics_.cuda_transfer_store_total_us);
  AtomicStoreZero(&metrics_.cuda_transfer_store_max_us);
  AtomicStoreZero(&metrics_.cuda_transfer_retrieve_total_us);
  AtomicStoreZero(&metrics_.cuda_transfer_retrieve_max_us);
  AtomicStoreZero(&metrics_.cuda_transfer_wait_event_total_us);
  AtomicStoreZero(&metrics_.cuda_transfer_open_tensors_total_us);
  AtomicStoreZero(&metrics_.cuda_transfer_copy_total_us);
  AtomicStoreZero(&metrics_.cuda_transfer_cache_total_us);
  AtomicStoreZero(&metrics_.cuda_transfer_completion_event_total_us);
  AtomicStoreZero(&metrics_.cuda_transfer_bytes);
  AtomicStoreZero(&metrics_.cuda_transfer_memcpy_calls);
  AtomicStoreZero(&metrics_.cuda_transfer_kernel_calls);
}


}  // namespace lmcache::mp
