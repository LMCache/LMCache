// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/protocol.h"

#include <array>
#include <cstring>
#include <string_view>

namespace lmcache::mp {

struct RequestTypeSpec {
  RequestType type;
  const char* name;
};

constexpr std::array<RequestTypeSpec, 23> kRequestTypes = {{
    {RequestType::kRegisterKvCache, "REGISTER_KV_CACHE"},
    {RequestType::kUnregisterKvCache, "UNREGISTER_KV_CACHE"},
    {RequestType::kStore, "STORE"},
    {RequestType::kRetrieve, "RETRIEVE"},
    {RequestType::kLookup, "LOOKUP"},
    {RequestType::kQueryPrefetchStatus, "QUERY_PREFETCH_STATUS"},
    {RequestType::kQueryPrefetchLookupHits, "QUERY_PREFETCH_LOOKUP_HITS"},
    {RequestType::kFreeLookupLocks, "FREE_LOOKUP_LOCKS"},
    {RequestType::kEndSession, "END_SESSION"},
    {RequestType::kClear, "CLEAR"},
    {RequestType::kGetChunkSize, "GET_CHUNK_SIZE"},
    {RequestType::kPing, "PING"},
    {RequestType::kReportBlockAllocation, "REPORT_BLOCK_ALLOCATION"},
    {RequestType::kNoop, "NOOP"},
    {RequestType::kCbRegisterKvCache, "CB_REGISTER_KV_CACHE"},
    {RequestType::kCbUnregisterKvCache, "CB_UNREGISTER_KV_CACHE"},
    {RequestType::kCbStorePreComputed, "CB_STORE_PRE_COMPUTED"},
    {RequestType::kCbLookupPreComputed, "CB_LOOKUP_PRE_COMPUTED"},
    {RequestType::kCbRetrievePreComputed, "CB_RETRIEVE_PRE_COMPUTED"},
    {RequestType::kCbStoreFinal, "CB_STORE_FINAL"},
    {RequestType::kCbLookupPreComputedV2, "CB_LOOKUP_PRE_COMPUTED_V2"},
    {RequestType::kCbRetrievePreComputedV2, "CB_RETRIEVE_PRE_COMPUTED_V2"},
    {RequestType::kLookupWithResult, "LOOKUP_WITH_RESULT"},
}};

std::string RequestTypeName(RequestType type) {
  for (const auto& spec : kRequestTypes) {
    if (spec.type == type) {
      return spec.name;
    }
  }
  return "UNKNOWN";
}

}  // namespace lmcache::mp

extern "C" {

std::uint32_t lmcache_mp_cpp_protocol_version() {
  return lmcache::mp::LMCACHE_MP_PROTOCOL_VERSION;
}

int lmcache_mp_cpp_request_type_value(const char* name, std::uint32_t* out) {
  if (name == nullptr || out == nullptr) {
    return -1;
  }
  for (const auto& spec : lmcache::mp::kRequestTypes) {
    if (std::strcmp(name, spec.name) == 0) {
      *out = static_cast<std::uint32_t>(spec.type);
      return 1;
    }
  }
  return 0;
}

const char* lmcache_mp_cpp_request_type_name(std::uint32_t value) {
  for (const auto& spec : lmcache::mp::kRequestTypes) {
    if (static_cast<std::uint32_t>(spec.type) == value) {
      return spec.name;
    }
  }
  return nullptr;
}
}
