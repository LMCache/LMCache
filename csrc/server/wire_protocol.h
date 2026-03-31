// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — msgpack wire protocol encode/decode
//
// Must produce bytes identical to Python `msgspec.msgpack` so that the
// existing Python vLLM clients work without modification.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "types.h"

// msgpack types are only needed in the .cpp (via pimpl); no forward
// declarations needed here.

namespace lmcache {
namespace server {

// ============================================================================
// Typed payload structs — decoded from inbound ZMQ frames
// ============================================================================

struct RegisterPayload {
  int32_t instance_id;
  std::vector<CudaIpcTensorDesc> kv_caches;
  std::string model_name;
  int32_t world_size;
};

struct StorePayload {
  IPCCacheEngineKey key;
  int32_t instance_id;
  std::vector<int32_t> gpu_block_ids;
  std::vector<uint8_t> event_ipc_handle;  // raw bytes
};

struct RetrievePayload {
  IPCCacheEngineKey key;
  int32_t instance_id;
  std::vector<int32_t> gpu_block_ids;
  std::vector<uint8_t> event_ipc_handle;
  int32_t skip_first_n_tokens;
};

struct LookupPayload {
  IPCCacheEngineKey key;
  int32_t tp_size;
};

struct FreeLookupLocksPayload {
  IPCCacheEngineKey key;
  int32_t tp_size;
};

struct EndSessionPayload {
  std::string request_id;
};

struct QueryPrefetchStatusPayload {
  int32_t prefetch_job_id;
};

struct QueryPrefetchLookupHitsPayload {
  int32_t prefetch_job_id;
};

// ============================================================================
// Encoder — serialises response types to msgpack bytes
// ============================================================================

class Encoder {
 public:
  Encoder();
  ~Encoder();

  /// Encode a RequestUID
  std::vector<uint8_t> encode_request_uid(RequestUID uid);

  /// Encode a RequestType enum value
  std::vector<uint8_t> encode_request_type(RequestType type);

  /// Encode a (event_ipc_handle, success) response pair
  std::vector<uint8_t> encode_store_response(
      const std::vector<uint8_t>& event_ipc_handle, bool success);

  /// Encode a (event_ipc_handle, success) response pair for retrieve
  std::vector<uint8_t> encode_retrieve_response(
      const std::vector<uint8_t>& event_ipc_handle, bool success);

  /// Encode an int response (lookup hit count, chunk_size, etc.)
  std::vector<uint8_t> encode_int_response(int64_t value);

  /// Encode a bool response (ping)
  std::vector<uint8_t> encode_bool_response(bool value);

  /// Encode a None response (for void handlers)
  std::vector<uint8_t> encode_none_response();

  /// Encode an optional int response (None or int)
  std::vector<uint8_t> encode_optional_int_response(int64_t value,
                                                    bool is_none);

  /// Encode a string response (debug)
  std::vector<uint8_t> encode_string_response(const std::string& value);

 private:
  struct Impl;
  Impl* impl_;
};

// ============================================================================
// Decoder — parses inbound ZMQ frames from msgpack bytes
// ============================================================================

class Decoder {
 public:
  Decoder();
  ~Decoder();

  /// Decode a RequestUID from raw bytes
  RequestUID decode_request_uid(const uint8_t* data, size_t len);

  /// Decode a RequestType enum value from raw bytes
  RequestType decode_request_type(const uint8_t* data, size_t len);

  /// Decode a RegisterPayload
  RegisterPayload decode_register_payload(
      const std::vector<std::vector<uint8_t>>& frames);

  /// Decode a StorePayload
  StorePayload decode_store_payload(
      const std::vector<std::vector<uint8_t>>& frames);

  /// Decode a RetrievePayload
  RetrievePayload decode_retrieve_payload(
      const std::vector<std::vector<uint8_t>>& frames);

  /// Decode a LookupPayload (also used for SYNC_LOOKUP)
  LookupPayload decode_lookup_payload(
      const std::vector<std::vector<uint8_t>>& frames);

  /// Decode a FreeLookupLocksPayload
  FreeLookupLocksPayload decode_free_lookup_locks_payload(
      const std::vector<std::vector<uint8_t>>& frames);

  /// Decode an EndSessionPayload
  EndSessionPayload decode_end_session_payload(
      const std::vector<std::vector<uint8_t>>& frames);

  /// Decode a QueryPrefetchStatusPayload
  QueryPrefetchStatusPayload decode_query_prefetch_status_payload(
      const std::vector<std::vector<uint8_t>>& frames);

  /// Decode a QueryPrefetchLookupHitsPayload
  QueryPrefetchLookupHitsPayload decode_query_prefetch_lookup_hits_payload(
      const std::vector<std::vector<uint8_t>>& frames);

  /// Decode an int payload (e.g., instance_id for unregister)
  int32_t decode_int_payload(const uint8_t* data, size_t len);

  /// Decode CudaIPCWrapper from Ext type code 1 (pickle or structured)
  CudaIpcTensorDesc decode_cuda_ipc_wrapper(const uint8_t* data, size_t len);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace server
}  // namespace lmcache
