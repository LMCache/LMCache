#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

struct LmcacheMpCppCache;

struct LmcacheMpCppStats {
  std::uint64_t dram_bytes;
  std::uint64_t disk_bytes;
  std::uint64_t dram_entries;
  std::uint64_t disk_entries;
  std::uint64_t total_entries;
  std::uint64_t locked_entries;
  std::uint64_t lock_count;
  std::uint64_t locked_bytes;
  std::uint64_t pinned_entries;
  std::uint64_t eviction_count;
};

LmcacheMpCppCache* lmcache_mp_cpp_cache_create(
    std::uint64_t dram_capacity_bytes, const char* disk_path);

void lmcache_mp_cpp_cache_destroy(LmcacheMpCppCache* cache);

int lmcache_mp_cpp_cache_put(LmcacheMpCppCache* cache, const char* key,
                             const std::uint8_t* data, std::uint64_t len);

int lmcache_mp_cpp_cache_get(LmcacheMpCppCache* cache, const char* key,
                             std::uint8_t* out, std::uint64_t len);

int lmcache_mp_cpp_cache_exists(LmcacheMpCppCache* cache, const char* key);

int lmcache_mp_cpp_cache_size(LmcacheMpCppCache* cache, const char* key,
                              std::uint64_t* out_len);

int lmcache_mp_cpp_cache_remove(LmcacheMpCppCache* cache, const char* key);

int lmcache_mp_cpp_cache_lock(LmcacheMpCppCache* cache, const char* key);

int lmcache_mp_cpp_cache_unlock(LmcacheMpCppCache* cache, const char* key);

int lmcache_mp_cpp_cache_pin(LmcacheMpCppCache* cache, const char* key);

int lmcache_mp_cpp_cache_unpin(LmcacheMpCppCache* cache, const char* key);

int lmcache_mp_cpp_cache_is_resident(LmcacheMpCppCache* cache, const char* key);

int lmcache_mp_cpp_cache_clear(LmcacheMpCppCache* cache);

int lmcache_mp_cpp_cache_clear_force(LmcacheMpCppCache* cache);

LmcacheMpCppStats lmcache_mp_cpp_cache_stats(LmcacheMpCppCache* cache);

const char* lmcache_mp_cpp_cache_last_error(LmcacheMpCppCache* cache);

int lmcache_mp_cpp_cache_last_error_copy(LmcacheMpCppCache* cache, char* out,
                                         std::uint64_t out_len,
                                         std::uint64_t* needed_len);
}
