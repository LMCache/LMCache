// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lmcache::mp {

struct IpcCacheEngineKey {
  std::string model_name;
  std::uint32_t world_size = 0;
  std::optional<std::uint32_t> worker_id;
  std::vector<std::uint32_t> token_ids;
  std::uint64_t start = 0;
  std::uint64_t end = 0;
  std::string request_id;
  std::string cache_salt;
};

std::optional<IpcCacheEngineKey> DecodeIpcCacheEngineKey(
    const std::uint8_t* data, std::uint64_t len, std::string* error);

std::optional<std::string> DecodeNoContextLookupRequestId(
    const std::uint8_t* data, std::uint64_t len, std::string* error);

std::vector<std::string> ObjectKeyStringsForIpcKey(const IpcCacheEngineKey& key,
                                                   std::uint64_t chunk_size,
                                                   std::uint64_t start,
                                                   std::uint64_t end,
                                                   std::string* error);

std::vector<std::string> LookupObjectKeyStringsForIpcKey(
    const IpcCacheEngineKey& key, std::uint64_t chunk_size, std::string* error);

}  // namespace lmcache::mp

extern "C" {

struct LmcacheMpCppIpcKeySummary {
  std::uint64_t token_count;
  std::uint64_t start;
  std::uint64_t end;
  std::uint64_t model_name_len;
  std::uint64_t request_id_len;
  std::uint64_t cache_salt_len;
  std::uint32_t world_size;
  std::int32_t worker_id;
};

int lmcache_mp_cpp_ipc_key_summary(const std::uint8_t* data, std::uint64_t len,
                                   LmcacheMpCppIpcKeySummary* out);

int lmcache_mp_cpp_decode_ipc_key(
    const std::uint8_t* data, std::uint64_t len, LmcacheMpCppIpcKeySummary* out,
    char* model_name, std::uint64_t model_name_len, char* request_id,
    std::uint64_t request_id_len, char* cache_salt,
    std::uint64_t cache_salt_len, std::uint32_t* token_ids,
    std::uint64_t max_token_ids);

int lmcache_mp_cpp_ipc_key_object_key_strings(
    const std::uint8_t* data, std::uint64_t len, std::uint64_t chunk_size,
    std::uint64_t start, std::uint64_t end, char* out, std::uint64_t out_len,
    std::uint64_t* needed_len, std::uint64_t* out_count);
}
