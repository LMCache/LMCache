// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/native_server.h"

#include "lmcache_mp_cpp/cuda_transfer.h"
#include "lmcache_mp_cpp/ipc_key.h"
#include "lmcache_mp_cpp/kv_metadata.h"
#include "lmcache_mp_cpp/msgpack_lite.h"
#include "lmcache_mp_cpp/protocol.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lmcache::mp {
namespace {

using msgpack::Bytes;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

void AtomicMax(std::atomic<std::uint64_t>* value, std::uint64_t candidate) {
  std::uint64_t current = value->load(std::memory_order_relaxed);
  while (current < candidate &&
         !value->compare_exchange_weak(current, candidate,
                                       std::memory_order_relaxed)) {
  }
}

std::uint64_t ElapsedMicros(std::chrono::steady_clock::time_point start) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

KvTensorMetadata TensorMetadataFromWrapper(
    const msgpack::DecodedCudaIpcWrapper& wrapper) {
  return KvTensorMetadata{
      .kind = wrapper.kind,
      .dtype = wrapper.dtype,
      .shape = wrapper.shape,
      .stride = wrapper.stride,
      .storage_offset = wrapper.storage_offset,
      .device_uuid = wrapper.device_uuid,
      .storage_bytes =
          wrapper.storage_bytes != 0 ? wrapper.storage_bytes
                                     : wrapper.raw_nbytes,
      .storage_offset_bytes = wrapper.storage_offset_bytes,
      .ipc_handle = wrapper.ipc_handle,
      .event_handle = wrapper.event_handle,
      .event_sync_required = wrapper.event_sync_required,
  };
}

std::vector<KvTensorMetadata> TensorMetadataFromWrappers(
    const std::vector<msgpack::DecodedCudaIpcWrapper>& wrappers) {
  std::vector<KvTensorMetadata> tensors;
  tensors.reserve(wrappers.size());
  for (const msgpack::DecodedCudaIpcWrapper& wrapper : wrappers) {
    tensors.push_back(TensorMetadataFromWrapper(wrapper));
  }
  return tensors;
}

}  // namespace

NativeServer::Frames NativeServer::HandleRequest(const RequestTask& task) {
  metrics_.request_count.fetch_add(1, std::memory_order_relaxed);
  Frames response = task.prefix;
  const auto type = static_cast<RequestType>(task.request_type);

  switch (type) {
    case RequestType::kPing:
      response.push_back(msgpack::EncodeBool(true));
      break;
    case RequestType::kGetChunkSize:
      response.push_back(msgpack::EncodeUnsigned(config_.chunk_size));
      break;
    case RequestType::kNoop:
      response.push_back(msgpack::EncodeString("OK"));
      break;
    case RequestType::kClear:
      ForceClearCache();
      break;
    case RequestType::kRegisterKvCache:
      HandleRegisterKvCache(task.payloads);
      break;
    case RequestType::kUnregisterKvCache:
      HandleUnregisterKvCache(task.payloads);
      break;
    case RequestType::kFreeLookupLocks:
      HandleFreeLookupLocks(task.payloads);
      break;
    case RequestType::kEndSession:
      HandleEndSession(task.payloads);
      break;
    case RequestType::kReportBlockAllocation:
      HandleReportBlockAllocation(task.payloads);
      break;
    case RequestType::kLookup:
      metrics_.lookup_count.fetch_add(1, std::memory_order_relaxed);
      (void)HandleLookupPayload(task.payloads, true);
      break;
    case RequestType::kLookupWithResult:
      metrics_.lookup_count.fetch_add(1, std::memory_order_relaxed);
      response.push_back(
          msgpack::EncodeUnsigned(HandleLookupPayload(task.payloads, false)));
      break;
    case RequestType::kQueryPrefetchStatus:
      response.push_back(
          msgpack::EncodeUnsigned(LookupResult(task.payloads, true)));
      break;
    case RequestType::kQueryPrefetchLookupHits:
      response.push_back(
          msgpack::EncodeUnsigned(LookupResult(task.payloads, false)));
      break;
    case RequestType::kStore:
      return HandleStorePayload(task);
    case RequestType::kRetrieve:
      return HandleRetrievePayload(task);
    case RequestType::kCbRegisterKvCache:
      (void)ValidateCbRegisterKvCache(task.payloads);
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      break;
    case RequestType::kCbUnregisterKvCache:
      (void)ValidateCbUnregisterKvCache(task.payloads);
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      break;
    case RequestType::kCbLookupPreComputed:
      (void)ValidateCbLookupPayload(task.payloads);
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      response.push_back(msgpack::EncodeEmptyArray());
      break;
    case RequestType::kCbLookupPreComputedV2:
      (void)ValidateCbLookupPayload(task.payloads);
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      response.push_back(msgpack::EncodeEmptyArray());
      break;
    case RequestType::kCbStorePreComputed:
      (void)ValidateCbStorePayload(task.payloads);
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      response.push_back(msgpack::EncodeBytesBoolTuple(Bytes{}, false));
      break;
    case RequestType::kCbRetrievePreComputed:
      (void)ValidateCbRetrievePayload(task.payloads, false);
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      response.push_back(msgpack::EncodeBytesBoolTuple(Bytes{}, false));
      break;
    case RequestType::kCbStoreFinal:
      (void)ValidateCbStorePayload(task.payloads);
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      response.push_back(msgpack::EncodeBytesBoolTuple(Bytes{}, false));
      break;
    case RequestType::kCbRetrievePreComputedV2:
      (void)ValidateCbRetrievePayload(task.payloads, true);
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      response.push_back(msgpack::EncodeBytesBoolTuple(Bytes{}, false));
      break;
    default:
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      std::cerr << "unsupported native MP request type "
                << static_cast<int>(task.request_type) << "\n";
      response.push_back(msgpack::EncodeNil());
      break;
  }
  return response;
}

void NativeServer::HandleRegisterKvCache(const Frames& payloads) {
  if (payloads.size() < 6) {
    (void)RecordInvalidPayload(
        "REGISTER_KV_CACHE missing required payload frames");
    std::cerr << "REGISTER_KV_CACHE missing required payload frames\n";
    return;
  }
  const auto instance_id = msgpack::DecodeUnsigned(payloads[0]);
  const auto kv_cache_len = msgpack::DecodeArraySizeHeader(payloads[1]);
  const auto model_name = msgpack::DecodeString(payloads[2]);
  const auto world_size = msgpack::DecodeUnsigned(payloads[3]);
  const auto engine_type = msgpack::DecodeString(payloads[4]);
  const auto layout_hints = msgpack::DecodeStringMap(payloads[5]);
  if (!instance_id || !kv_cache_len || !model_name || !world_size ||
      *world_size == 0 ||
      *world_size > std::numeric_limits<std::uint32_t>::max() || !engine_type ||
      !layout_hints) {
    (void)RecordInvalidPayload("failed to decode REGISTER_KV_CACHE metadata");
    std::cerr << "failed to decode REGISTER_KV_CACHE metadata\n";
    return;
  }
  std::optional<std::vector<msgpack::DecodedCudaIpcWrapper>> kv_wrappers;
  if (*kv_cache_len != 0) {
    kv_wrappers = msgpack::DecodeCudaIpcWrapperArray(payloads[1]);
    if (!kv_wrappers) {
      (void)RecordInvalidPayload(
          "failed to decode native-friendly KV cache metadata");
      std::cerr << "failed to decode native-friendly KV cache metadata\n";
      return;
    }
  }
  if (*engine_type != "vllm" && *engine_type != "sglang" &&
      *engine_type != "trtllm" && *engine_type != "mock") {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "unsupported REGISTER_KV_CACHE engine_type " << *engine_type
              << "\n";
    return;
  }

  std::string kv_layout;
  std::uint32_t logical_block_size = 0;
  if (const auto it = layout_hints->find("kv_layout");
      it != layout_hints->end()) {
    if (it->second.kind != msgpack::DecodedValue::Kind::kString ||
        (it->second.string_value != "NHD" &&
         it->second.string_value != "HND")) {
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      std::cerr << "invalid REGISTER_KV_CACHE kv_layout hint\n";
      return;
    }
    kv_layout = it->second.string_value;
  }
  if (const auto it = layout_hints->find("inference_engine_logical_block_size");
      it != layout_hints->end()) {
    if (it->second.kind != msgpack::DecodedValue::Kind::kUnsigned ||
        it->second.unsigned_value == 0 ||
        it->second.unsigned_value > std::numeric_limits<std::uint32_t>::max()) {
      metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
      std::cerr << "invalid REGISTER_KV_CACHE logical block-size hint\n";
      return;
    }
    logical_block_size = static_cast<std::uint32_t>(it->second.unsigned_value);
  }
  std::string unsupported_error;
  std::optional<std::uint64_t> compress_ratio_hint;
  if (!ValidateSupportedLayoutHints(*layout_hints, &compress_ratio_hint,
                                    &unsupported_error)) {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    std::cerr << unsupported_error << "\n";
    return;
  }
  bool use_layerwise_hint = false;
  for (const std::string name : {"use_layerwise", "layerwise"}) {
    if (const msgpack::DecodedValue* hint = FindHint(*layout_hints, name)) {
      bool disabled = true;
      if (DecodeDisabledHint(*hint, &disabled) && !disabled) {
        use_layerwise_hint = true;
      }
    }
  }
  TrtLlmLayoutHints trt_llm_hints;
  if (!DecodeTrtLlmLayoutHints(*layout_hints, *engine_type, kv_layout,
                               &trt_llm_hints, &unsupported_error)) {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    std::cerr << unsupported_error << "\n";
    return;
  }
  if (trt_llm_hints.present && kv_layout.empty()) {
    kv_layout = "HND";
  }
  std::optional<std::vector<std::uint64_t>> group_physical_block_sizes;
  if (!DecodeOptionalUnsignedArrayHint(
          *layout_hints, "group_physical_block_sizes",
          &group_physical_block_sizes, &unsupported_error)) {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    std::cerr << unsupported_error << "\n";
    return;
  }
  std::optional<std::vector<std::uint64_t>> group_compress_ratios;
  if (!DecodeOptionalUnsignedArrayHint(*layout_hints, "group_compress_ratios",
                                       &group_compress_ratios,
                                       &unsupported_error)) {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    std::cerr << unsupported_error << "\n";
    return;
  }
  if (logical_block_size == 0 && kv_wrappers && !kv_wrappers->empty()) {
    const auto shape_info =
        InferKvShapeInfo(kv_wrappers->front().shape, kv_layout, trt_llm_hints);
    if (shape_info) {
      if (shape_info->block_size > std::numeric_limits<std::uint32_t>::max()) {
        metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "registered KV block-size is too large\n";
        return;
      }
      logical_block_size = static_cast<std::uint32_t>(shape_info->block_size);
    }
  }
  if (kv_wrappers &&
      !ValidateRegisteredKvWrappersSupported(
          *kv_wrappers, kv_layout, logical_block_size, trt_llm_hints,
          compress_ratio_hint, group_physical_block_sizes,
          group_compress_ratios, &unsupported_error)) {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    std::cerr << unsupported_error << "\n";
    return;
  }
  if (!kv_wrappers &&
      ((compress_ratio_hint && *compress_ratio_hint != 1) ||
       trt_llm_hints.present ||
       (group_physical_block_sizes && !group_physical_block_sizes->empty()) ||
       (group_compress_ratios && !group_compress_ratios->empty()))) {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "native REGISTER_KV_CACHE requires KV wrapper metadata for "
                 "compressed or TRT-LLM layouts\n";
    return;
  }

  std::vector<KvTensorMetadata> warmup_tensors;
  RegisteredContext context = {
      .model_name = *model_name,
      .world_size = static_cast<std::uint32_t>(*world_size),
      .engine_type = *engine_type,
      .kv_layout = kv_layout,
      .inference_engine_logical_block_size = logical_block_size,
      .use_layerwise_hint = use_layerwise_hint,
      .trt_llm_layout_hints = trt_llm_hints.present,
      .trt_llm_num_kv_heads = trt_llm_hints.num_kv_heads,
      .trt_llm_tokens_per_block = trt_llm_hints.tokens_per_block,
      .trt_llm_head_dim = trt_llm_hints.head_dim,
      .kv_cache_wrapper_count = static_cast<std::uint64_t>(*kv_cache_len),
  };
  if (kv_wrappers && !kv_wrappers->empty()) {
    const auto& first = kv_wrappers->front();
    context.first_kv_dtype = first.dtype;
    context.first_kv_shape = first.shape;
    context.first_kv_stride = first.stride;
    context.first_kv_device_uuid = first.device_uuid;
    context.first_kv_ipc_handle_bytes = first.ipc_handle_bytes;
    context.first_kv_storage_bytes =
        first.storage_bytes != 0 ? first.storage_bytes : first.raw_nbytes;
    context.first_kv_storage_offset_bytes = first.storage_offset_bytes;
    context.first_kv_event_handle_bytes = first.event_handle_bytes;
    context.first_kv_event_sync_required = first.event_sync_required;
    if (auto shape_info =
            InferKvShapeInfo(first.shape, kv_layout, trt_llm_hints)) {
      context.first_kv_num_blocks = shape_info->num_blocks;
      context.first_kv_block_size = shape_info->block_size;
    }
    if (config_.enable_cuda_gpu_hot_cache) {
      warmup_tensors = TensorMetadataFromWrappers(*kv_wrappers);
    }
    context.kv_wrappers = std::move(*kv_wrappers);
  }
  {
    std::lock_guard<std::mutex> lock(registered_contexts_mu_);
    registered_contexts_[*instance_id] = std::move(context);
  }
  LaunchCudaTensorWarmup(std::move(warmup_tensors));
}

void NativeServer::HandleUnregisterKvCache(const Frames& payloads) {
  if (payloads.empty()) {
    (void)RecordInvalidPayload(
        "UNREGISTER_KV_CACHE missing required payload frames");
    return;
  }
  const auto instance_id = msgpack::DecodeUnsigned(payloads[0]);
  if (!instance_id) {
    (void)RecordInvalidPayload(
        "failed to decode UNREGISTER_KV_CACHE instance id");
    return;
  }
  std::lock_guard<std::mutex> lock(registered_contexts_mu_);
  registered_contexts_.erase(*instance_id);
}

void NativeServer::HandleReportBlockAllocation(const Frames& payloads) {
  if (payloads.size() < 3) {
    (void)RecordInvalidPayload(
        "REPORT_BLOCK_ALLOCATION missing required payload frames");
    std::cerr << "REPORT_BLOCK_ALLOCATION missing required payload frames\n";
    return;
  }
  const auto instance_id = msgpack::DecodeUnsigned(payloads[0]);
  const auto model_name = msgpack::DecodeString(payloads[1]);
  const auto records = msgpack::DecodeBlockAllocationRecords(payloads[2]);
  if (!instance_id || !model_name || !records) {
    (void)RecordInvalidPayload(
        "failed to decode REPORT_BLOCK_ALLOCATION payload");
    std::cerr << "failed to decode REPORT_BLOCK_ALLOCATION payload\n";
    return;
  }

  metrics_.block_allocation_report_count.fetch_add(1,
                                                   std::memory_order_relaxed);
  metrics_.block_allocation_record_count.fetch_add(records->size(),
                                                   std::memory_order_relaxed);

  BlockAllocationSummary summary;
  summary.valid = true;
  summary.instance_id = *instance_id;
  summary.model_name = *model_name;
  summary.record_count = records->size();
  if (!records->empty()) {
    const auto& last = records->back();
    summary.last_request_id = last.req_id;
    summary.last_new_block_count = last.new_block_ids.size();
    summary.last_new_token_count = last.new_token_ids.size();
  }
  std::lock_guard<std::mutex> lock(block_allocations_mu_);
  last_block_allocation_ = std::move(summary);
}

bool NativeServer::ValidateCbRegisterKvCache(const Frames& payloads) {
  if (payloads.size() < 4) {
    return RecordInvalidPayload(
        "CB_REGISTER_KV_CACHE missing required payload frames");
  }
  const auto instance_id = msgpack::DecodeUnsigned(payloads[0]);
  const auto kv_cache_len = msgpack::DecodeArraySizeHeader(payloads[1]);
  const auto model_name = msgpack::DecodeString(payloads[2]);
  const auto world_size = msgpack::DecodeUnsigned(payloads[3]);
  if (!instance_id || !kv_cache_len || !model_name || !world_size ||
      *world_size == 0) {
    return RecordInvalidPayload(
        "failed to decode CB_REGISTER_KV_CACHE payload");
  }
  return true;
}

bool NativeServer::ValidateCbUnregisterKvCache(const Frames& payloads) {
  if (payloads.empty()) {
    return RecordInvalidPayload(
        "CB_UNREGISTER_KV_CACHE missing required payload frames");
  }
  if (!msgpack::DecodeUnsigned(payloads[0])) {
    return RecordInvalidPayload(
        "failed to decode CB_UNREGISTER_KV_CACHE payload");
  }
  return true;
}

bool NativeServer::ValidateCbLookupPayload(const Frames& payloads) {
  if (payloads.empty()) {
    return RecordInvalidPayload("CB_LOOKUP missing IPCCacheEngineKey payload");
  }
  std::string error;
  const auto key =
      DecodeIpcCacheEngineKey(payloads[0].data(), payloads[0].size(), &error);
  if (!key) {
    return RecordInvalidPayload(
        "failed to decode CB_LOOKUP IPCCacheEngineKey: " + error);
  }
  (void)LookupObjectKeyStringsForIpcKey(*key, config_.chunk_size, &error);
  if (!error.empty()) {
    return RecordInvalidPayload("failed to expand CB_LOOKUP ObjectKeys: " +
                                error);
  }
  return true;
}

bool NativeServer::ValidateCbStorePayload(const Frames& payloads) {
  if (payloads.size() < 4) {
    return RecordInvalidPayload("CB_STORE missing required payload frames");
  }
  std::string error;
  const auto key =
      DecodeIpcCacheEngineKey(payloads[0].data(), payloads[0].size(), &error);
  const auto offset = msgpack::DecodeUnsigned(payloads[1]);
  const auto instance_id = msgpack::DecodeUnsigned(payloads[2]);
  const auto event_handle = msgpack::DecodeBinary(payloads[3]);
  if (!key || !offset || !instance_id || !event_handle) {
    if (!error.empty()) {
      return RecordInvalidPayload("failed to decode CB_STORE payload: " +
                                  error);
    }
    return RecordInvalidPayload("failed to decode CB_STORE payload");
  }
  return true;
}

bool NativeServer::ValidateCbRetrievePayload(const Frames& payloads, bool v2) {
  if (payloads.size() < 5) {
    return RecordInvalidPayload("CB_RETRIEVE missing required payload frames");
  }
  std::string error;
  const auto key =
      DecodeIpcCacheEngineKey(payloads[0].data(), payloads[0].size(), &error);
  const bool ranges_ok =
      v2 ? msgpack::DecodeCbMatchResults(payloads[1]).has_value()
         : msgpack::DecodeUnsignedPairArray(payloads[1]).has_value();
  const auto offset = msgpack::DecodeUnsigned(payloads[2]);
  const auto instance_id = msgpack::DecodeUnsigned(payloads[3]);
  const auto event_handle = msgpack::DecodeBinary(payloads[4]);
  if (!key || !ranges_ok || !offset || !instance_id || !event_handle) {
    if (!error.empty()) {
      return RecordInvalidPayload("failed to decode CB_RETRIEVE payload: " +
                                  error);
    }
    return RecordInvalidPayload("failed to decode CB_RETRIEVE payload");
  }
  return true;
}

bool NativeServer::HasRegisteredContext(const IpcCacheEngineKey& key) const {
  std::lock_guard<std::mutex> lock(registered_contexts_mu_);
  for (const auto& entry : registered_contexts_) {
    const RegisteredContext& context = entry.second;
    if (context.model_name == key.model_name &&
        context.world_size == key.world_size) {
      return true;
    }
  }
  return false;
}

std::uint64_t NativeServer::HandleLookupPayload(const Frames& payloads,
                                                bool store_result) {
  if (payloads.empty()) {
    (void)RecordInvalidPayload("LOOKUP missing IPCCacheEngineKey payload");
    std::cerr << "LOOKUP missing IPCCacheEngineKey payload\n";
    return 0;
  }

  std::string error;
  auto key =
      DecodeIpcCacheEngineKey(payloads[0].data(), payloads[0].size(), &error);
  if (!key) {
    (void)RecordInvalidPayload("failed to decode LOOKUP IPCCacheEngineKey: " +
                               error);
    std::cerr << "failed to decode LOOKUP IPCCacheEngineKey: " << error << "\n";
    return 0;
  }

  if (!HasRegisteredContext(*key)) {
    if (store_result) {
      std::lock_guard<std::mutex> lock(lookup_results_mu_);
      lookup_results_[key->request_id] = 0;
    }
    return 0;
  }

  std::vector<std::string> object_keys =
      LookupObjectKeyStringsForIpcKey(*key, config_.chunk_size, &error);
  if (!error.empty()) {
    (void)RecordInvalidPayload("failed to expand LOOKUP ObjectKeys: " + error);
    std::cerr << "failed to expand LOOKUP ObjectKeys: " << error << "\n";
    return 0;
  }

  ReleaseLocksForRequest(key->request_id);

  std::uint64_t hit_objects = 0;
  std::uint64_t l1_hits = 0;
  std::uint64_t l2_hits = 0;
  std::uint64_t l2_misses = 0;
  std::uint64_t hot_cache_hits = 0;
  std::vector<std::string> locked_object_keys;
  for (const std::string& object_key : object_keys) {
    const int exists = lmcache_mp_cpp_cache_exists(cache_, object_key.c_str());
    if (exists == 1) {
      const int locked = lmcache_mp_cpp_cache_lock(cache_, object_key.c_str());
      if (locked == 1) {
        ++hit_objects;
        ++l1_hits;
        locked_object_keys.push_back(object_key);
      }
      continue;
    }
    if (config_.enable_cuda_gpu_hot_cache &&
        CudaTransferDeviceChunkReady(object_key)) {
      ++hit_objects;
      ++hot_cache_hits;
      continue;
    }
    bool found_in_l2 = false;
    {
      std::lock_guard<std::mutex> l2_lock(l2_adapters_mu_);
      for (const auto& adapter : l2_adapters_) {
        if (adapter->Exists(object_key)) {
          found_in_l2 = true;
          break;
        }
      }
    }
    if (found_in_l2) {
      ++hit_objects;
      ++l2_hits;
    } else if (!l2_adapters_.empty()) {
      ++l2_misses;
    }
  }
  const std::uint64_t missed_objects =
      object_keys.size() > hit_objects ? object_keys.size() - hit_objects : 0;
  metrics_.cache_hits.fetch_add(hit_objects, std::memory_order_relaxed);
  metrics_.cache_misses.fetch_add(missed_objects, std::memory_order_relaxed);
  metrics_.l1_hit_count.fetch_add(l1_hits + hot_cache_hits,
                                  std::memory_order_relaxed);
  metrics_.l2_hit_count.fetch_add(l2_hits, std::memory_order_relaxed);
  metrics_.l2_miss_count.fetch_add(l2_misses, std::memory_order_relaxed);
  if (hit_objects > 0 && missed_objects > 0) {
    metrics_.partial_hit_count.fetch_add(1, std::memory_order_relaxed);
  }

  const std::uint64_t divisor = key->worker_id ? 1 : key->world_size;
  const std::uint64_t hit_chunks = divisor == 0 ? 0 : hit_objects / divisor;
  {
    std::lock_guard<std::mutex> lock(lookup_results_mu_);
    if (store_result) {
      lookup_results_[key->request_id] = hit_chunks;
    }
    if (!locked_object_keys.empty()) {
      lookup_locks_[key->request_id] = std::move(locked_object_keys);
    }
  }
  return hit_chunks;
}

void NativeServer::HandleFreeLookupLocks(const Frames& payloads) {
  if (payloads.size() < 2) {
    (void)RecordInvalidPayload("FREE_LOOKUP_LOCKS missing payload frames");
    std::cerr << "FREE_LOOKUP_LOCKS missing payload frames\n";
    return;
  }

  std::string error;
  auto key =
      DecodeIpcCacheEngineKey(payloads[0].data(), payloads[0].size(), &error);
  if (!key) {
    (void)RecordInvalidPayload(
        "failed to decode FREE_LOOKUP_LOCKS IPCCacheEngineKey: " + error);
    std::cerr << "failed to decode FREE_LOOKUP_LOCKS IPCCacheEngineKey: "
              << error << "\n";
    return;
  }
  (void)msgpack::DecodeUnsigned(payloads[1]);
  std::vector<std::string> object_keys = ObjectKeyStringsForIpcKey(
      *key, config_.chunk_size, key->start, key->end, &error);
  if (!error.empty()) {
    (void)RecordInvalidPayload(
        "failed to expand FREE_LOOKUP_LOCKS ObjectKeys: " + error);
    std::cerr << "failed to expand FREE_LOOKUP_LOCKS ObjectKeys: " << error
              << "\n";
    return;
  }
  ReleaseLocksForRequestKeys(key->request_id, object_keys);
}

void NativeServer::HandleEndSession(const Frames& payloads) {
  if (payloads.empty()) {
    (void)RecordInvalidPayload("END_SESSION missing request id payload");
    return;
  }
  const auto request_id = msgpack::DecodeString(payloads[0]);
  if (!request_id) {
    (void)RecordInvalidPayload("failed to decode END_SESSION request id");
    return;
  }
  std::vector<std::string> object_keys;
  {
    std::lock_guard<std::mutex> lock(lookup_results_mu_);
    lookup_results_.erase(*request_id);
    auto it = lookup_locks_.find(*request_id);
    if (it != lookup_locks_.end()) {
      object_keys = std::move(it->second);
      lookup_locks_.erase(it);
    }
  }
  if (!object_keys.empty()) {
    ReleaseLocks(object_keys);
  }
}

std::optional<NativeServer::KvTransferPlan> NativeServer::BuildKvTransferPlan(
    const Frames& payloads, bool retrieve) {
  const std::size_t expected_payloads = retrieve ? 5 : 4;
  if (payloads.size() < expected_payloads) {
    (void)RecordInvalidPayload(std::string(retrieve ? "RETRIEVE" : "STORE") +
                               " missing required payload frames");
    return std::nullopt;
  }

  std::string error;
  auto key =
      DecodeIpcCacheEngineKey(payloads[0].data(), payloads[0].size(), &error);
  const auto instance_id = msgpack::DecodeUnsigned(payloads[1]);
  const auto gpu_block_ids = msgpack::DecodeUnsignedArray(payloads[2]);
  const auto event_handle = msgpack::DecodeBinary(payloads[3]);
  if (!key || !instance_id || !gpu_block_ids || !event_handle) {
    std::string message = std::string("failed to decode ") +
                          (retrieve ? "RETRIEVE" : "STORE") +
                          " payload metadata";
    if (!error.empty()) {
      message += ": " + error;
    }
    (void)RecordInvalidPayload(message);
    return std::nullopt;
  }
  std::uint64_t skip_first_n_tokens = 0;
  if (retrieve) {
    const auto maybe_skip = msgpack::DecodeUnsigned(payloads[4]);
    if (!maybe_skip) {
      (void)RecordInvalidPayload(
          "failed to decode RETRIEVE skip_first_n_tokens");
      return std::nullopt;
    }
    skip_first_n_tokens = *maybe_skip;
  }
  if (!key->worker_id) {
    (void)RecordInvalidPayload(std::string(retrieve ? "RETRIEVE" : "STORE") +
                               " requires worker_id");
    return std::nullopt;
  }

  RegisteredContext context;
  {
    std::lock_guard<std::mutex> lock(registered_contexts_mu_);
    const auto it = registered_contexts_.find(*instance_id);
    if (it == registered_contexts_.end()) {
      (void)RecordInvalidPayload(std::string(retrieve ? "RETRIEVE" : "STORE") +
                                 " references unregistered instance");
      return std::nullopt;
    }
    context = it->second;
  }
  if (context.model_name != key->model_name ||
      context.world_size != key->world_size) {
    (void)RecordInvalidPayload(std::string(retrieve ? "RETRIEVE" : "STORE") +
                               " metadata does not match registered context");
    return std::nullopt;
  }
  if (context.inference_engine_logical_block_size == 0) {
    (void)RecordInvalidPayload(
        "registered context missing logical block-size metadata");
    return std::nullopt;
  }
  if (config_.chunk_size % context.inference_engine_logical_block_size != 0) {
    (void)RecordInvalidPayload(
        "chunk size is not divisible by registered logical block size");
    return std::nullopt;
  }
  if (context.first_kv_block_size != 0 &&
      (context.first_kv_block_size >
           context.inference_engine_logical_block_size ||
       context.inference_engine_logical_block_size %
               context.first_kv_block_size !=
           0)) {
    (void)RecordInvalidPayload(
        "registered KV block-size metadata does not divide layout hint");
    return std::nullopt;
  }

  std::vector<std::string> object_keys = ObjectKeyStringsForIpcKey(
      *key, config_.chunk_size, key->start, key->end, &error);
  if (!error.empty()) {
    (void)RecordInvalidPayload(std::string("failed to expand ") +
                               (retrieve ? "RETRIEVE" : "STORE") +
                               " ObjectKeys: " + error);
    return std::nullopt;
  }
  if (object_keys.empty() && key->start != key->end) {
    (void)RecordInvalidPayload(std::string(retrieve ? "RETRIEVE" : "STORE") +
                               " has no chunk keys for non-empty token range");
    return std::nullopt;
  }
  if (retrieve &&
      skip_first_n_tokens > object_keys.size() * config_.chunk_size) {
    (void)RecordInvalidPayload(
        "RETRIEVE skip_first_n_tokens exceeds requested token range");
    return std::nullopt;
  }
  if (retrieve &&
      skip_first_n_tokens % context.inference_engine_logical_block_size != 0) {
    (void)RecordInvalidPayload(
        "RETRIEVE skip_first_n_tokens is not block aligned");
    return std::nullopt;
  }

  const std::uint64_t blocks_per_chunk =
      config_.chunk_size / context.inference_engine_logical_block_size;
  const std::uint64_t expected_blocks =
      static_cast<std::uint64_t>(object_keys.size()) * blocks_per_chunk;
  if (gpu_block_ids->size() != expected_blocks) {
    (void)RecordInvalidPayload(std::string(retrieve ? "RETRIEVE" : "STORE") +
                               " gpu_block_ids length does not match token "
                               "range and logical block size");
    return std::nullopt;
  }
  if (context.first_kv_num_blocks != 0) {
    for (std::uint64_t block_id : *gpu_block_ids) {
      if (block_id >= context.first_kv_num_blocks) {
        (void)RecordInvalidPayload(
            std::string(retrieve ? "RETRIEVE" : "STORE") +
            " gpu_block_id is outside registered KV cache block range");
        return std::nullopt;
      }
    }
  }
  return KvTransferPlan{
      .key = std::move(*key),
      .context = std::move(context),
      .instance_id = *instance_id,
      .gpu_block_ids = std::move(*gpu_block_ids),
      .event_handle = std::move(*event_handle),
      .object_keys = std::move(object_keys),
      .skip_first_n_tokens = skip_first_n_tokens,
      .blocks_per_chunk = blocks_per_chunk,
  };
}

bool NativeServer::ValidateKvTransferPayload(const Frames& payloads,
                                             bool retrieve) {
  return BuildKvTransferPlan(payloads, retrieve).has_value();
}

bool NativeServer::WriteStoredChunksToL2(
    const std::vector<std::string>& object_keys) {
  if (l2_adapters_.empty()) {
    return true;
  }
  bool ok = true;
  for (const std::string& object_key : object_keys) {
    std::uint64_t stored_size = 0;
    if (lmcache_mp_cpp_cache_size(cache_, object_key.c_str(), &stored_size) !=
        1) {
      metrics_.l2_error_count.fetch_add(1, std::memory_order_relaxed);
      std::cerr << "cannot write missing native cache key to L2: " << object_key
                << "\n";
      ok = false;
      continue;
    }
    std::vector<std::uint8_t> bytes(stored_size);
    if (lmcache_mp_cpp_cache_get(cache_, object_key.c_str(), bytes.data(),
                                 bytes.size()) != 1) {
      metrics_.l2_error_count.fetch_add(1, std::memory_order_relaxed);
      const char* cache_error = lmcache_mp_cpp_cache_last_error(cache_);
      std::cerr << "cannot read native cache key for L2 write: " << object_key
                << ": " << (cache_error == nullptr ? "" : cache_error) << "\n";
      ok = false;
      continue;
    }
    std::lock_guard<std::mutex> l2_lock(l2_adapters_mu_);
    for (const auto& adapter : l2_adapters_) {
      std::string error;
      if (!adapter->Put(object_key, bytes, &error)) {
        metrics_.l2_error_count.fetch_add(1, std::memory_order_relaxed);
        std::cerr << "native L2 write failed for " << object_key << ": "
                  << error << "\n";
        ok = false;
        continue;
      }
      metrics_.l2_store_count.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return ok;
}

void NativeServer::LoadMissingChunksFromL2(
    const std::vector<std::string>& object_keys) {
  if (l2_adapters_.empty()) {
    return;
  }
  for (const std::string& object_key : object_keys) {
    if (lmcache_mp_cpp_cache_exists(cache_, object_key.c_str()) == 1) {
      continue;
    }

    std::optional<std::vector<std::uint8_t>> found;
    {
      std::lock_guard<std::mutex> l2_lock(l2_adapters_mu_);
      for (const auto& adapter : l2_adapters_) {
        std::string error;
        auto bytes = adapter->Get(object_key, &error);
        if (!error.empty()) {
          metrics_.l2_error_count.fetch_add(1, std::memory_order_relaxed);
          std::cerr << "native L2 read failed for " << object_key << ": "
                    << error << "\n";
          continue;
        }
        if (bytes) {
          found = std::move(bytes);
          break;
        }
      }
    }
    if (!found) {
      continue;
    }
    if (lmcache_mp_cpp_cache_put(cache_, object_key.c_str(), found->data(),
                                 found->size()) != 1) {
      metrics_.l2_error_count.fetch_add(1, std::memory_order_relaxed);
      const char* cache_error = lmcache_mp_cpp_cache_last_error(cache_);
      std::cerr << "native cache put failed during L2 load for " << object_key
                << ": " << (cache_error == nullptr ? "" : cache_error) << "\n";
      continue;
    }
    metrics_.l2_load_count.fetch_add(1, std::memory_order_relaxed);
  }
}

void NativeServer::ClearL2Adapters() {
  if (l2_adapters_.empty()) {
    return;
  }
  std::lock_guard<std::mutex> l2_lock(l2_adapters_mu_);
  for (const auto& adapter : l2_adapters_) {
    std::string error;
    if (!adapter->Clear(&error)) {
      metrics_.l2_error_count.fetch_add(1, std::memory_order_relaxed);
      std::cerr << "native L2 clear failed: " << error << "\n";
    }
  }
}

bool NativeServer::LockTransferChunks(
    const std::vector<std::string>& object_keys,
    std::vector<std::string>* locked_object_keys, std::uint64_t* lock_epoch,
    const std::string& context) {
  const auto wait_start = std::chrono::steady_clock::now();
  *lock_epoch = force_clear_epoch_.load(std::memory_order_acquire);
  locked_object_keys->clear();
  locked_object_keys->reserve(object_keys.size());
  for (const std::string& object_key : object_keys) {
    const int locked = lmcache_mp_cpp_cache_lock(cache_, object_key.c_str());
    if (locked == 1) {
      locked_object_keys->push_back(object_key);
      metrics_.transfer_lock_count.fetch_add(1, std::memory_order_relaxed);
      continue;
    }

    metrics_.transfer_lock_failure_count.fetch_add(1,
                                                   std::memory_order_relaxed);
    const char* cache_error = lmcache_mp_cpp_cache_last_error(cache_);
    std::cerr << "failed to lock native cache key for " << context << ": "
              << object_key << ": "
              << (cache_error == nullptr ? "" : cache_error) << "\n";
    ReleaseLocksForEpoch(*locked_object_keys, *lock_epoch);
    locked_object_keys->clear();
    const std::uint64_t wait_us = ElapsedMicros(wait_start);
    metrics_.transfer_lock_wait_total_us.fetch_add(
        wait_us, std::memory_order_relaxed);
    AtomicMax(&metrics_.transfer_lock_wait_max_us, wait_us);
    return false;
  }
  const std::uint64_t wait_us = ElapsedMicros(wait_start);
  metrics_.transfer_lock_wait_total_us.fetch_add(wait_us,
                                                 std::memory_order_relaxed);
  AtomicMax(&metrics_.transfer_lock_wait_max_us, wait_us);
  return true;
}

NativeServer::Frames NativeServer::HandleStorePayload(const RequestTask& task) {
  metrics_.store_count.fetch_add(1, std::memory_order_relaxed);
  Frames response = task.prefix;
  auto plan = BuildKvTransferPlan(task.payloads, false);
  if (!plan) {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    response.push_back(msgpack::EncodeBytesBoolTuple(Bytes{}, false));
    return response;
  }

  KvTransferRequest request;
  request.object_keys = plan->object_keys;
  request.gpu_block_ids = plan->gpu_block_ids;
  request.input_event_handle = plan->event_handle;
  request.chunk_size = config_.chunk_size;
  request.logical_block_size =
      plan->context.inference_engine_logical_block_size;
  request.kv_layout = plan->context.kv_layout;
  request.trt_llm_layout_hints = plan->context.trt_llm_layout_hints;
  request.trt_llm_num_kv_heads = plan->context.trt_llm_num_kv_heads;
  request.trt_llm_tokens_per_block = plan->context.trt_llm_tokens_per_block;
  request.trt_llm_head_dim = plan->context.trt_llm_head_dim;
  request.enable_gpu_hot_cache = config_.enable_cuda_gpu_hot_cache;
  request.tensors = TensorMetadataFromWrappers(plan->context.kv_wrappers);

  bool store_materialized_in_cpu_cache = true;
  KvTransferResult result;
  if (config_.enable_cuda_gpu_hot_cache) {
    result = StoreKvChunksToCudaHotCacheAsync(request);
    store_materialized_in_cpu_cache = false;
    if (!result.success) {
      std::cerr << "native async hot-cache STORE failed; falling back to "
                   "CPU-cache STORE: "
                << result.error << "\n";
      result = StoreKvChunksFromCuda(request, cache_);
      store_materialized_in_cpu_cache = true;
    }
  } else {
    result = StoreKvChunksFromCuda(request, cache_);
  }
  RecordCudaTransferStats(false, result.stats);
  if (!result.success) {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "native STORE CUDA transfer failed: " << result.error << "\n";
  } else if (store_materialized_in_cpu_cache) {
    std::vector<std::string> transfer_locks;
    std::uint64_t transfer_lock_epoch = 0;
    if (!LockTransferChunks(request.object_keys, &transfer_locks,
                            &transfer_lock_epoch, "STORE L2 write")) {
      result.success = false;
      result.completion_event_handle.clear();
    } else {
      const auto lock_hold_start = std::chrono::steady_clock::now();
      if (!WriteStoredChunksToL2(request.object_keys)) {
        result.success = false;
        result.completion_event_handle.clear();
      }
      ReleaseLocksForEpoch(transfer_locks, transfer_lock_epoch);
      const std::uint64_t hold_us = ElapsedMicros(lock_hold_start);
      metrics_.transfer_lock_hold_total_us.fetch_add(
          hold_us, std::memory_order_relaxed);
      AtomicMax(&metrics_.transfer_lock_hold_max_us, hold_us);
    }
  }
  response.push_back(msgpack::EncodeBytesBoolTuple(
      result.completion_event_handle, result.success));
  return response;
}

NativeServer::Frames NativeServer::HandleRetrievePayload(
    const RequestTask& task) {
  metrics_.retrieve_count.fetch_add(1, std::memory_order_relaxed);
  Frames response = task.prefix;
  auto plan = BuildKvTransferPlan(task.payloads, true);
  if (!plan) {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    response.push_back(msgpack::EncodeBytesBoolTuple(Bytes{}, false));
    return response;
  }

  KvTransferRequest request;
  request.object_keys = plan->object_keys;
  request.gpu_block_ids = plan->gpu_block_ids;
  request.input_event_handle = plan->event_handle;
  request.chunk_size = config_.chunk_size;
  request.logical_block_size =
      plan->context.inference_engine_logical_block_size;
  request.skip_first_n_tokens = plan->skip_first_n_tokens;
  request.kv_layout = plan->context.kv_layout;
  request.trt_llm_layout_hints = plan->context.trt_llm_layout_hints;
  request.trt_llm_num_kv_heads = plan->context.trt_llm_num_kv_heads;
  request.trt_llm_tokens_per_block = plan->context.trt_llm_tokens_per_block;
  request.trt_llm_head_dim = plan->context.trt_llm_head_dim;
  request.enable_gpu_hot_cache = config_.enable_cuda_gpu_hot_cache;
  request.tensors = TensorMetadataFromWrappers(plan->context.kv_wrappers);

  KvTransferResult result;
  const bool hot_cache_ready =
      config_.enable_cuda_gpu_hot_cache &&
      std::all_of(request.object_keys.begin(), request.object_keys.end(),
                  [](const std::string& key) {
                    return CudaTransferDeviceChunkReady(key);
                  });
  if (hot_cache_ready) {
    result = RetrieveKvChunksToCuda(request, cache_);
    RecordCudaTransferStats(true, result.stats);
  } else {
    LoadMissingChunksFromL2(request.object_keys);
    std::vector<std::string> transfer_locks;
    std::uint64_t transfer_lock_epoch = 0;
    if (!LockTransferChunks(request.object_keys, &transfer_locks,
                            &transfer_lock_epoch, "RETRIEVE")) {
      result.success = false;
      result.error = "failed to lock native cache chunks for RETRIEVE";
    } else {
      const auto lock_hold_start = std::chrono::steady_clock::now();
      result = RetrieveKvChunksToCuda(request, cache_);
      RecordCudaTransferStats(true, result.stats);
      ReleaseLocksForEpoch(transfer_locks, transfer_lock_epoch);
      const std::uint64_t hold_us = ElapsedMicros(lock_hold_start);
      metrics_.transfer_lock_hold_total_us.fetch_add(
          hold_us, std::memory_order_relaxed);
      AtomicMax(&metrics_.transfer_lock_hold_max_us, hold_us);
    }
  }
  if (!result.success) {
    metrics_.unsupported_count.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "native RETRIEVE CUDA transfer failed: " << result.error
              << "\n";
  }
  response.push_back(msgpack::EncodeBytesBoolTuple(
      result.completion_event_handle, result.success));
  return response;
}

bool NativeServer::RecordInvalidPayload(const std::string& message) {
  metrics_.invalid_payload_count.fetch_add(1, std::memory_order_relaxed);
  std::cerr << message << "\n";
  return false;
}

void NativeServer::ForceClearCache() {
  force_clear_epoch_.fetch_add(1, std::memory_order_acq_rel);
  lmcache_mp_cpp_cache_clear_force(cache_);
  ClearCudaTransferDeviceCache();
  ClearL2Adapters();
  force_clear_epoch_.fetch_add(1, std::memory_order_acq_rel);
  {
    std::lock_guard<std::mutex> lock(lookup_results_mu_);
    lookup_results_.clear();
    lookup_locks_.clear();
  }
  metrics_.clear_count.fetch_add(1, std::memory_order_relaxed);
}

void NativeServer::ReleaseLocks(const std::vector<std::string>& object_keys) {
  ReleaseLocksForEpoch(object_keys,
                       force_clear_epoch_.load(std::memory_order_acquire));
}

void NativeServer::ReleaseLocksForEpoch(
    const std::vector<std::string>& object_keys, std::uint64_t lock_epoch) {
  for (const std::string& object_key : object_keys) {
    const int unlocked =
        lmcache_mp_cpp_cache_unlock(cache_, object_key.c_str());
    if (unlocked < 0) {
      const std::uint64_t current_epoch =
          force_clear_epoch_.load(std::memory_order_acquire);
      if (current_epoch != lock_epoch || current_epoch % 2 != 0) {
        continue;
      }
      const char* cache_error = lmcache_mp_cpp_cache_last_error(cache_);
      std::cerr << "failed to release native cache lock for " << object_key
                << ": " << (cache_error == nullptr ? "" : cache_error) << "\n";
    }
  }
}

void NativeServer::ReleaseLocksForRequest(const std::string& request_id) {
  std::vector<std::string> object_keys;
  const std::uint64_t lock_epoch =
      force_clear_epoch_.load(std::memory_order_acquire);
  {
    std::lock_guard<std::mutex> lock(lookup_results_mu_);
    auto it = lookup_locks_.find(request_id);
    if (it == lookup_locks_.end()) {
      return;
    }
    object_keys = std::move(it->second);
    lookup_locks_.erase(it);
  }
  ReleaseLocksForEpoch(object_keys, lock_epoch);
}

void NativeServer::ReleaseLocksForRequestKeys(
    const std::string& request_id,
    const std::vector<std::string>& object_keys) {
  std::vector<std::string> release_keys;
  const std::uint64_t lock_epoch =
      force_clear_epoch_.load(std::memory_order_acquire);
  {
    std::lock_guard<std::mutex> lock(lookup_results_mu_);
    auto it = lookup_locks_.find(request_id);
    if (it == lookup_locks_.end()) {
      return;
    }
    std::vector<std::string>& held_keys = it->second;
    for (const std::string& object_key : object_keys) {
      auto held_it = std::find(held_keys.begin(), held_keys.end(), object_key);
      if (held_it == held_keys.end()) {
        continue;
      }
      release_keys.push_back(*held_it);
      held_keys.erase(held_it);
    }
    if (held_keys.empty()) {
      lookup_locks_.erase(it);
    }
  }
  ReleaseLocksForEpoch(release_keys, lock_epoch);
}

std::uint64_t NativeServer::LookupResult(const Frames& payloads,
                                         bool erase_result) {
  if (payloads.empty()) {
    (void)RecordInvalidPayload("QUERY_PREFETCH missing request id payload");
    return 0;
  }
  const auto request_id = msgpack::DecodeString(payloads[0]);
  if (!request_id) {
    (void)RecordInvalidPayload("failed to decode QUERY_PREFETCH request id");
    return 0;
  }

  std::lock_guard<std::mutex> lock(lookup_results_mu_);
  const auto it = lookup_results_.find(*request_id);
  if (it == lookup_results_.end()) {
    return 0;
  }
  const std::uint64_t result = it->second;
  if (erase_result) {
    lookup_results_.erase(it);
  }
  return result;
}

}  // namespace lmcache::mp
