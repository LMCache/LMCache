// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>

namespace lmcache::mp {

constexpr std::uint32_t LMCACHE_MP_PROTOCOL_VERSION = 1;

enum class RequestType : std::uint8_t {
  kRegisterKvCache = 1,
  kUnregisterKvCache = 2,
  kStore = 3,
  kRetrieve = 4,
  kLookup = 5,
  kQueryPrefetchStatus = 6,
  kQueryPrefetchLookupHits = 7,
  kFreeLookupLocks = 8,
  kEndSession = 9,
  kClear = 10,
  kGetChunkSize = 11,
  kPing = 12,
  kReportBlockAllocation = 13,
  kNoop = 14,
  kCbRegisterKvCache = 15,
  kCbUnregisterKvCache = 16,
  kCbStorePreComputed = 17,
  kCbLookupPreComputed = 18,
  kCbRetrievePreComputed = 19,
  kCbStoreFinal = 20,
  kCbLookupPreComputedV2 = 21,
  kCbRetrievePreComputedV2 = 22,
  kLookupWithResult = 23,
};

std::string RequestTypeName(RequestType type);

}  // namespace lmcache::mp

extern "C" {

std::uint32_t lmcache_mp_cpp_protocol_version();

int lmcache_mp_cpp_request_type_value(const char* name, std::uint32_t* out);

const char* lmcache_mp_cpp_request_type_name(std::uint32_t value);
}
