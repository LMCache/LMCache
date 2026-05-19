// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lmcache::mp {

struct L2AdapterConfig {
  std::string type;
  std::string base_path;
};

struct L2AdapterStatus {
  std::string type;
  std::string base_path;
  std::uint64_t stored_files = 0;
  std::uint64_t stored_bytes = 0;
};

class L2Adapter {
 public:
  virtual ~L2Adapter() = default;

  virtual bool Put(const std::string& object_key,
                   const std::vector<std::uint8_t>& bytes,
                   std::string* error) = 0;
  virtual std::optional<std::vector<std::uint8_t>> Get(
      const std::string& object_key, std::string* error) = 0;
  virtual bool Delete(const std::string& object_key, std::string* error) = 0;
  virtual bool Clear(std::string* error) = 0;
  virtual bool Exists(const std::string& object_key) const = 0;
  virtual L2AdapterStatus Status() const = 0;
  virtual std::unordered_map<std::string, std::uint64_t> UsageBytesByCacheSalt()
      const = 0;
};

std::optional<L2AdapterConfig> ParseL2AdapterConfig(const std::string& json,
                                                    std::string* error);

std::unique_ptr<L2Adapter> CreateL2Adapter(const L2AdapterConfig& config,
                                           std::string* error);

std::string ObjectKeyStringToFsFilename(const std::string& object_key);

}  // namespace lmcache::mp

extern "C" {

struct LmcacheMpCppL2Adapter;

LmcacheMpCppL2Adapter* lmcache_mp_cpp_fs_l2_create(const char* base_path);

void lmcache_mp_cpp_fs_l2_destroy(LmcacheMpCppL2Adapter* adapter);

int lmcache_mp_cpp_fs_l2_put(LmcacheMpCppL2Adapter* adapter, const char* key,
                             const std::uint8_t* data, std::uint64_t len);

int lmcache_mp_cpp_fs_l2_size(LmcacheMpCppL2Adapter* adapter, const char* key,
                              std::uint64_t* out_len);

int lmcache_mp_cpp_fs_l2_get(LmcacheMpCppL2Adapter* adapter, const char* key,
                             std::uint8_t* out, std::uint64_t len);

int lmcache_mp_cpp_fs_l2_delete(LmcacheMpCppL2Adapter* adapter,
                                const char* key);

int lmcache_mp_cpp_fs_l2_clear(LmcacheMpCppL2Adapter* adapter);

int lmcache_mp_cpp_fs_l2_exists(LmcacheMpCppL2Adapter* adapter,
                                const char* key);

const char* lmcache_mp_cpp_fs_l2_last_error(LmcacheMpCppL2Adapter* adapter);

int lmcache_mp_cpp_fs_l2_filename(const char* key, char* out,
                                  std::uint64_t out_len,
                                  std::uint64_t* needed_len);
}
